"""
Script for sampling images on perturbations based on PC of variational prompt encodings
"""

import torch as th
from torchvision import transforms
import json

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)

SEED = 0
th.manual_seed(SEED)

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

prompt = "A plate with a brownie and frosting"
# prompt = "A zebra grazing on lush green grass in a field."
# prompt = "a small airplane that is on a runway"
batch_size = 1
guidance_scale = 3.0
full_batch_size = batch_size * 2

# Tune this parameter to control the sharpness of 256x256 images.
# A value of 1.0 is sharper, but sometimes results in grainy artifacts.
upsample_temp = 0.997

# Create a classifier-free guidance sampling function
def model_fn(x_t, ts, **kwargs):
    half = x_t[: len(x_t) // 2]
    combined = th.cat([half, half], dim=0)
    model_out = model(combined, ts, **kwargs)
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
    eps = th.cat([half_eps, half_eps], dim=0)
    return th.cat([eps, rest], dim=1)

def get_encoding(prompt : str):
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
    text_outputs = model.get_text_emb(tokens, mask)
    return text_outputs

orig_text_outputs = get_encoding(prompt)

# alpha represents target loss over batch of 32
alpha = 1.0

pdist = th.nn.MSELoss()
downsample = transforms.Resize(size=[32, 32])

# load prompts and generate pca for each one
with open("prompt_brownie_variations.json", 'r') as f:
    prompt_variations = json.load(f)

for i, p_var in prompt_variations:
    print(p_var)
    var_text_outputs = get_encoding(p_var)
    pc = th.load(f"brownie_variations/brownie_variation{i}_pca.pt")

    outputs = []
    for i in range(var_text_outputs['xf_out'].shape[-1]):
        th.manual_seed(SEED)
        th.cuda.manual_seed(SEED)

        zeros_mask = th.zeros_like(var_text_outputs['xf_out'][0])
        #loss = pdist(var_text_outputs['xf_out'][0], orig_text_outputs['xf_out'][0])
        scale_out = 1.0
        dir_out = pc[i].unsqueeze(0).repeat(512, 1)
        dir_out = th.cat((dir_out.unsqueeze(0), zeros_mask.unsqueeze(0)), 0).to(th.float16)
        xf_out = var_text_outputs['xf_out'] + dir_out / scale_out * alpha

        #loss = pdist(var_text_outputs['xf_proj'][0], text_outputs['xf_proj'][0])
        scale_proj = 1.0
        dir_proj = var_text_outputs['xf_proj'] - orig_text_outputs['xf_proj']
        xf_proj = orig_text_outputs['xf_proj'] + dir_proj / scale_proj * alpha

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

        outputs.append(downsample(samples[0]))

    outputs = th.stack(outputs).flatten(start_dim=1)
    # outputs = ((outputs+1) * 127.5).round().clamp(0, 255).permute(1,0).to(th.float32)
    th.save(outputs, f"brownie_variations_outputs/brownie_variation{i}_pca_output.pt")
