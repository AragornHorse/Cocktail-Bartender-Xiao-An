import numpy as np
import torch
import config
import random


datas = np.load(config.mat_pth)


def get_auto_encoder_datasets():
    idx = list(range(datas.shape[0]))
    random.shuffle(idx)

    test_idx = idx[:config.auto_encoder.test_num]
    train_idx = idx[config.auto_encoder.test_num:]

    train_data = datas[train_idx]
    test_data = datas[test_idx]

    return (
        torch.tensor(train_data, dtype=torch.float32, device=config.device),
        torch.tensor(test_data, dtype=torch.float32, device=config.device)
    )


def get_diffusion_datasets():
    idx = list(range(datas.shape[0]))
    random.shuffle(idx)

    test_idx = idx[:config.diffusion.test_num]
    train_idx = idx[config.diffusion.test_num:]

    train_data = datas[train_idx]
    test_data = datas[test_idx]

    return (
        torch.tensor(train_data, dtype=torch.float32, device=config.device),
        torch.tensor(test_data, dtype=torch.float32, device=config.device)
    )


def transform(x, strength=0.1):
    noise = torch.rand_like(x)
    exist = x > 0
    noise[~exist] = 0.
    noise[exist] = 1. + 2 * strength * noise[exist] - strength
    x = x * noise
    return x / torch.sum(x, dim=1, keepdim=True)


if __name__ == '__main__':
    x = torch.tensor([
        [0.1, 0.9, 0., 0.],
        [0.2, 0., 0.4, 0.4]
    ])
    for i in range(10):
        print(transform(x, 0.1))





