
import torch

def get_loss_fx():
    return LogCosh


def LogCosh(output, target):
    return torch.mean(torch.log(torch.cosh(output-target)))

