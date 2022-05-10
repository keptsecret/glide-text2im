from PIL import Image

import torch as th

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)

has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')

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

stuff_to_save = ["transformer", "transformer_proj", "final_ln", "token_embedding", "positional_embedding", "padding_embedding"]
transformer_only_dict = {}
for param_tensor in model.state_dict():
    for stuff in stuff_to_save:
        if stuff in param_tensor:
            transformer_only_dict[param_tensor] = model.state_dict()[param_tensor]
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

th.save(transformer_only_dict, "./transformer_only_weights.pt")