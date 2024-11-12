import torch

mat_pth = r"D:\Users\DELL\Desktop\datasets\cocktail\npy\mat.npy"

name2idx_pth = r"D:\Users\DELL\Desktop\datasets\cocktail\npy\name2idx.json"

ct2idx_pth = r"D:\Users\DELL\Desktop\datasets\cocktail\npy\ct2idx.json"

description_pth = r"D:\Users\DELL\Desktop\datasets\cocktail\npy\describe.npy"

latent_size = 48

device = torch.device("cuda")

debug = True


class AutoEncoderConfig:
    def __init__(self):
        self.hidden_sizes = [96, latent_size]
        self.dropout = 0.1
        self.path = r"D:\Users\DELL\PycharmProjects\pythonProject\cocktail\parameters\v1\autoencoder.pth"
        self.test_num = 50
        self.reg = 1e-3
        self.reg_func = 'kl'
        self.strength = 0.02


class DiffusionConfig:
    def __init__(self):
        self.hidden_sizes = [latent_size, 1024, 512, 256]
        self.dropout = 0.05
        self.T = 600
        self.temb_size = latent_size
        self.path = r"D:\Users\DELL\PycharmProjects\pythonProject\cocktail\parameters\v1\diffusion.pth"
        self.test_num = 50
        self.strength = 0.01


auto_encoder = AutoEncoderConfig()
diffusion = DiffusionConfig()



