import torch
import config
import torch.nn as nn
import torch.optim as optim
import datasets
import model


def mse_reg(latent):
    return torch.mean(latent ** 2)


def kl_reg(latent):
    mu = torch.mean(latent)
    sigma = torch.mean((latent - mu) ** 2)
    return torch.log(sigma) + (1 + mu ** 2) / sigma


model = model.AutoEncoder()
model.to(config.device)
opt = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
train_data, test_data = datasets.get_auto_encoder_datasets()
rec_loss_func = nn.CrossEntropyLoss()

for epoch in range(10000):
    x = datasets.transform(train_data, config.auto_encoder.strength)
    recon, latent = model(x)

    rec_loss = rec_loss_func(recon, x)

    if config.auto_encoder.reg_func == 'kl':
        reg_loss = kl_reg(latent)
    else:
        reg_loss = mse_reg(latent)

    loss = rec_loss + config.auto_encoder.reg * reg_loss

    opt.zero_grad()
    loss.backward()
    opt.step()

    print(f"epoch: {epoch}, loss: {rec_loss}")


model.eval()
with torch.no_grad():
    pred, latent = model(test_data)
    loss = rec_loss_func(pred, test_data)
print(f"eval, loss: {loss}")

model.save()

# sample_pred = torch.softmax(pred[0, :], dim=0)
# sample_gt = test_data[0, :]
#
# for i in range(sample_gt.shape[0]):
#     print(f"{sample_pred[i]}, {sample_gt[i]}")

