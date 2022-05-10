from PIL import Image
import torch as th
import numpy as np
from torchvision import transforms

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)
from glide_text2im.image_encoder import SketchEncoder, VQSketchEncoder

th.manual_seed(0)

# stuff for vq encoder ONLY
embed_dim = 256
n_embed = 16384
ddconfig = {
    "double_z": False,
    "z_channels": 256,
    "resolution": 256,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 128,
    "ch_mult": [1, 1, 2, 2, 4],
    "num_res_blocks": 2,
    "attn_resolutions": [16],
    "dropout": 0.0
}

has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')
# th.set_default_tensor_type('torch.cuda.FloatTensor')

# Create base model.
options = model_and_diffusion_defaults()
options['use_fp16'] = has_cuda
options['timestep_respacing'] = '100' # use 100 diffusion steps for fast sampling
model, diffusion = create_model_and_diffusion(**options)
model.eval()
if has_cuda:
    model.convert_to_fp16()
model.to(device)
model.load_state_dict(load_checkpoint('base', device))
print('total base parameters', sum(x.numel() for x in model.parameters()))

# Create upsampler model.
options_up = model_and_diffusion_defaults_upsampler()
options_up['use_fp16'] = has_cuda
options_up['timestep_respacing'] = 'fast27' # use 27 diffusion steps for very fast sampling
model_up, diffusion_up = create_model_and_diffusion(**options_up)
model_up.eval()
if has_cuda:
    model_up.convert_to_fp16()
model_up.to(device)
model_up.load_state_dict(load_checkpoint('upsample', device))
print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))

def show_images(batch: th.Tensor, filename):
    """ Display a batch of images inline. """
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
    # reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    # display(Image.fromarray(reshaped.numpy()))
    im = transforms.ToPILImage()(scaled)
    im.save(filename)

# set up image
im = Image.open("./000000000092.png")
img = transforms.ToTensor()(im)
img = img.reshape(img.shape[1:])
max_side = max(img.shape)
pad_left, pad_top = max_side-img.shape[1], max_side-img.shape[0]
padding = (pad_left//2, pad_top//2, pad_left//2+pad_left%2, pad_top//2+pad_top%2)
img = transforms.Pad(padding, fill=1)(img)
img = img.unsqueeze(0)
img = transforms.Resize(256)(img)
img = img.repeat(3, 1, 1).to(device=device)

img_batch = img.unsqueeze(0).cpu()

# sketch_encoder = SketchEncoder()
sketch_encoder = VQSketchEncoder(ddconfig=ddconfig, n_embed=n_embed, embed_dim=embed_dim, ckpt_path="vq_encoder_weights.pt")
sketch_encoder.load_state_dict(th.load("./sketch_encoder_weights_vq_f100mse_noh.pt"))
sketch_out, _, _ = sketch_encoder(img_batch)
sketch_tokens = model.get_sketch_emb(sketch_out.to(device))

prompt = "A white plate with a brownie and white frosting"
# prompt = "A zebra grazing on lush green grass in a field."
batch_size = 1
guidance_scale = 3.0
full_batch_size = batch_size * 2

# Tune this parameter to control the sharpness of 256x256 images.
# A value of 1.0 is sharper, but sometimes results in grainy artifacts.
upsample_temp = 0.997

# for testing perturbations
tokens = model.tokenizer.encode(prompt)
tokens, mask = model.tokenizer.padded_tokens_and_mask(
    tokens, options['text_ctx']
)

# Create the classifier-free guidance tokens (empty)
full_batch_size = batch_size * 2
uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
    [], options['text_ctx']
)

tokens = th.tensor([tokens] * batch_size + [uncond_tokens] * batch_size, device=device)
mask = th.tensor([mask] * batch_size + [uncond_mask] * batch_size, dtype=th.bool, device=device)

# alpha represents target loss over batch of 32
alpha = 1e-1
text_outputs = model.get_text_emb(tokens, mask)
# print(text_outputs['xf_out'].shape)
pdist = th.nn.MSELoss()

def model_fn(x_t, ts, **kwargs):
    half = x_t[: len(x_t) // 2]
    combined = th.cat([half, half], dim=0)
    model_out = model(combined, ts, **kwargs)
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
    eps = th.cat([half_eps, half_eps], dim=0)
    return th.cat([eps, rest], dim=1)

# loop through and perturb each vector individually
outputs = []
for vec_idx in range(sketch_tokens['xf_out'].shape[-1]):
    out_mask = th.zeros_like(sketch_tokens['xf_out'])
    out_mask[0, :, vec_idx] += 1
    loss = pdist(sketch_tokens['xf_out'][0], text_outputs['xf_out'][0])
    scale_out = loss.item()
    dir_out = sketch_tokens['xf_out'] - text_outputs['xf_out']
    xf_out = text_outputs['xf_out'] + out_mask * dir_out / scale_out * alpha

    loss = pdist(sketch_tokens['xf_proj'][0], text_outputs['xf_proj'][0])
    scale_proj = loss.item()
    dir_proj = sketch_tokens['xf_proj'] - text_outputs['xf_proj']
    xf_proj = text_outputs['xf_proj'] + dir_proj / scale_proj * alpha

    model_kwargs = dict(
        tokens=dict(xf_proj=xf_proj, xf_out=xf_out),
        mask=None,
    )

    # Sample from the base model.
    model.del_cache()
    samples = diffusion.p_sample_loop(
        model_fn,
        (full_batch_size, 3, options["image_size"], options["image_size"]),
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:batch_size]
    model.del_cache()
    outputs.append(samples[0])

def pca(X : th.Tensor):
    X = X - X.mean()
    Z = X / X.std()
    Z = th.matmul(Z.T, Z)
    L, V = th.linalg.eig(Z)
    return L, V

# use NxN eigenvectors as mask, correlates to how much to scale/perturb each input dimension by -- scales correlated inputs along with it
outputs = th.stack(outputs).flatten(start_dim=1)
th.save(outputs, "brownie_outputs.pt")
L, V = pca(outputs)
print(V.shape)
th.save(V, "brownie_pca.pt")