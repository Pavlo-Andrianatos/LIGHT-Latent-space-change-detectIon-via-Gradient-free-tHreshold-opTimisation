import torch
import torch.nn as nn
import numpy as np


def conv_trans_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2, padding=0),
    )
    return model


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_block_2(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn
    )
    return model


def difference_revised(tensor1, tensor2, threshold=1):
    temp_tensor1 = tensor1.cpu().numpy()
    temp_tensor2 = tensor2.cpu().numpy()

    mask = np.abs(temp_tensor1 - temp_tensor2) > threshold

    new_tensor = torch.from_numpy(temp_tensor1 * mask)

    new_tensor = new_tensor.float().cuda(0)

    return new_tensor
