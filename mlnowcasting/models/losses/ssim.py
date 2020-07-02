
import torch.nn as nn
from ...utils.torch_ssim import ssim


def get_loss_fx():
    return nn.MSELoss()

   
def SSIM(output, target):
    return 1-ssim(output[:,0,...], target[:,0,...], size_average=1, data_range=1)

