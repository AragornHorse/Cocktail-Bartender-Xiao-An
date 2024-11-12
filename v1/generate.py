import torch
import config
import model
import json

use_diffusion = True

auto_encoder = model.AutoEncoder()
auto_encoder.to(config.device)
auto_encoder.load()
auto_encoder.eval()


if use_diffusion:
    unet = model.UNet()
    unet.to(config.device)
    unet.load()
    unet.eval()

diffusion = model.Diffusion()

with torch.no_grad():

    if use_diffusion:
        gen_latent = diffusion.generate(unet, add_noise=True)

        print(torch.mean(torch.abs(gen_latent)))

    else:
        gen_latent = torch.randn(10, config.latent_size, device=config.device)

    gen_cocktail = auto_encoder.decode(gen_latent)
    gen_cocktail = torch.softmax(gen_cocktail, dim=-1).detach().cpu().numpy()


with open(config.ct2idx_pth, 'r', encoding='utf-8') as f:
    ct_to_idx = json.load(f)
    idx_to_ct = list(ct_to_idx.keys())

for cocktail in gen_cocktail:
    for ct_idx, ct_num in enumerate(cocktail):
        if ct_num > 0.05:
            print(f"{idx_to_ct[ct_idx]}: {ct_num}", end=', ')
    print("\n")

