import torch
import torch.nn as nn
import numpy as np
import config


class TimeEmbedding(nn.Module):
    def __init__(self):
        d_model = config.diffusion.temb_size
        T = config.diffusion.T
        assert d_model % 2 == 0
        super().__init__()

        self.timembedding = nn.Sequential(
            nn.Embedding(config.diffusion.T, d_model)
        )

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.cts = torch.tensor(
            np.load(config.description_pth), dtype=torch.float32, device=config.device, requires_grad=True
        )
        self.ct_fc = nn.Sequential(
            nn.Linear(self.cts.shape[1], config.auto_encoder.hidden_sizes[0], bias=True)
        )

        now_size = config.auto_encoder.hidden_sizes[0]
        lst = []
        for size in config.auto_encoder.hidden_sizes[1:]:
            lst.append(nn.Sequential(
                nn.BatchNorm1d(now_size),
                nn.Dropout(config.auto_encoder.dropout),
                nn.Linear(now_size, size, bias=True),
                nn.GELU()
            ))
            now_size = size
        self.encoder = nn.Sequential(*lst)

        lst = []
        for size in reversed(config.auto_encoder.hidden_sizes[:-1]):
            lst.append(nn.Sequential(
                nn.BatchNorm1d(now_size),
                nn.Dropout(config.auto_encoder.dropout),
                nn.Linear(now_size, size, bias=True),
                nn.GELU()
            ))
            now_size = size
        self.decoder = nn.Sequential(*lst)

    def forward(self, x):
        cts = self.ct_fc(self.cts)
        ct = x @ cts
        latent = self.encoder(ct)
        re_ct = self.decoder(latent)
        reconstruct = re_ct @ cts.transpose(0, 1)
        return reconstruct, latent

    def encode(self, x):
        cts = self.ct_fc(self.cts)
        ct = x @ cts
        latent = self.encoder(ct)
        return latent

    def decode(self, x):
        cts = self.ct_fc(self.cts)
        re_ct = self.decoder(x)
        reconstruct = re_ct @ cts.transpose(0, 1)
        return reconstruct

    def save(self):
        torch.save(self.state_dict(), config.auto_encoder.path)

    def load(self):
        self.load_state_dict(torch.load(config.auto_encoder.path))


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        lst = []
        t_lst = []
        now_size = config.diffusion.hidden_sizes[0]
        for size in config.diffusion.hidden_sizes[1:]:
            lst.append(nn.Sequential(
                nn.BatchNorm1d(now_size),
                nn.Dropout(config.diffusion.dropout),
                nn.Linear(now_size, size),
                nn.GELU()
            ))
            t_lst.append(nn.Sequential(
                nn.Linear(config.diffusion.temb_size, now_size, bias=False)
            ))
            now_size = size
        self.encoder = nn.ModuleList(lst)
        self.temb = TimeEmbedding()
        self.t_fc = nn.ModuleList(t_lst)

        self.head = nn.Sequential(
            nn.Linear(config.latent_size, config.diffusion.hidden_sizes[0]),
            nn.GELU(),
        )

        self.tail = nn.Sequential(
            nn.BatchNorm1d(config.diffusion.hidden_sizes[0]),
            nn.Linear(config.diffusion.hidden_sizes[0], config.diffusion.hidden_sizes[0]),
            nn.GELU(),
            nn.BatchNorm1d(config.diffusion.hidden_sizes[0]),
            nn.Linear(config.diffusion.hidden_sizes[0], config.latent_size)
        )

        lst = []
        for size in reversed(config.diffusion.hidden_sizes[:-1]):
            lst.append((nn.Sequential(
                nn.BatchNorm1d(now_size),
                nn.Dropout(config.diffusion.dropout),
                nn.Linear(now_size, size),
                nn.GELU()
            )))
            now_size = size

        self.decoder = nn.ModuleList(lst)

    def forward(self, x, t):
        temb = self.temb(t)
        h = self.head(x)
        hs = []
        for i, layer in enumerate(self.encoder):
            h = layer(h + self.t_fc[i](temb))
            hs.append(h)

        for i, layer in enumerate(self.decoder):
            h = layer((h + hs.pop()) * 0.5) + self.t_fc[-i-1](temb)

        h = self.tail(h)
        return h

    def save(self):
        torch.save(self.state_dict(), config.diffusion.path)

    def load(self):
        self.load_state_dict(torch.load(config.diffusion.path))


class Diffusion:
    def __init__(self):
        self.betas = torch.linspace(0.007, 0.025, config.diffusion.T, device=config.device)
        self.alphas = 1 - self.betas
        self.alphas_mult = torch.clone(self.alphas)
        for i in range(1, self.alphas_mult.shape[0]):
            self.alphas_mult[i] *= self.alphas_mult[i - 1]

    def pollute(self, x, t=None):

        if isinstance(t, int):
            t = torch.full([x.shape[0]], t)
        if t is None:
            t = torch.randint(0, config.diffusion.T, [x.shape[0]])

        t = t.to(x.device)
        noise = torch.randn_like(x)  # n, h
        t_ = t.reshape(-1).long()
        alpha = self.alphas_mult[t_].reshape(-1, 1)
        x_ = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise

        return x_, noise, t

    def generate(self, model, x=None, add_noise=False):
        model.eval()

        if x is None:
            x = torch.randn(8, config.latent_size, device=config.device)

        with torch.no_grad():
            for t in reversed(range(config.diffusion.T)):
                z = model(x, torch.full([x.shape[0]], t).long().to(x.device))

                if config.debug:
                    print(f"z: {torch.mean(torch.abs(z))}")

                if t > 1 and add_noise:
                    x = 1 / torch.sqrt(self.alphas[t]) * (x - (1 - self.alphas[t]) /
                                                          torch.sqrt(1 - self.alphas_mult[t]) * z) + \
                        torch.sqrt(
                            (1 - self.alphas[t]) * (1 - self.alphas_mult[t - 1]) /
                            (1 - self.alphas_mult[t])) * torch.randn_like(x)
                else:
                    x = 1 / torch.sqrt(self.alphas[t]) * \
                        (x - (1 - self.alphas[t]) / torch.sqrt(1 - self.alphas_mult[t]) * z)
                del z

        return x


if __name__ == '__main__':
    # model = AutoEncoder().to(config.device)
    #
    # x = torch.randn(32, 165).to(config.device)
    # rst = model(x)
    # print(rst[0].shape, rst[1].shape)
    # print(model.encode(x).shape)
    #
    # print(model.decode(rst[1]).shape)

    model = UNet()
    x = torch.randn(32, config.latent_size)

    print(model(x, torch.zeros(32).long()).shape)

    diffusion = Diffusion()

    print(diffusion.alphas_mult ** 0.5)
    pass

