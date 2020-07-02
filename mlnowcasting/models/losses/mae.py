
import torch

def get_loss_fx():
    return mae

def mae(input, target):
    return torch.mean(torch.abs((input - target)))

