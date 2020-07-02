
import torch
from ...utils.torch_ssim import ssim

def get_loss_fx():
    return combined

   
def SSIM(output, target):
    return 1-ssim(output[:,0,...], target[:,0,...], size_average=1, data_range=1)

def mse_loss(input, target):
    return torch.mean((input - target) ** 2)

def mae_loss(input, target):
    return torch.mean(torch.abs((input - target)))

def combined(output, target):
    return 0.2*SSIM(output, target) + mse_loss(output, target) + mae_loss(output, target)*0.1

