import torch
import torch.nn as nn
import torch.optim as optim
import model
import datasets
import config

train_data, test_data = datasets.get_diffusion_datasets()

auto_encoder = model.AutoEncoder()
auto_encoder.to(config.device)
auto_encoder.load()
auto_encoder.eval()

with torch.no_grad():
    train_data = auto_encoder.encode(train_data)
    test_data = auto_encoder.encode(test_data)

unet = model.UNet()
unet.to(config.device)
unet.load()
diffusion = model.Diffusion()

opt = optim.AdamW(unet.parameters(), lr=1e-3, weight_decay=1e-4)

loss_func = nn.L1Loss()

for epoch in range(10000):
    x, z, t = diffusion.pollute(train_data)

    z_hat = unet(x, t)

    loss = loss_func(z_hat, z)

    opt.zero_grad()
    loss.backward()
    opt.step()

    print(f"epoch: {epoch}, loss: {loss}")

unet.eval()
with torch.no_grad():
    x, z, t = diffusion.pollute(test_data)
    z_hat = unet(x, t)
    loss = loss_func(z_hat, z)

print(f"eval, loss: {loss}")

unet.save()
