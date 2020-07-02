# balanced MSE + MAE
import torch.nn as nn
import torch
import numpy as np

def get_loss_fx():
    return BMSE

    
BALANCING_WEIGHTS = (1, 1, 2, 5, 10, 30)
CENTRAL_REGION = (120, 120, 360, 360)
THRESHOLDS = np.array([0.5, 2, 5, 10, 30])
def rainfall2pixel(r, a=300, b=1.4):
    z_ = np.log10(r)*b*10+10*np.log10(a)
    return np.invert(np.round(np.array(z_)/70*255).astype('uint8'))/255

THRESHOLDS = rainfall2pixel(THRESHOLDS, a=300, b=1.4)

class Weighted_mse_mae(nn.Module):
    def __init__(self, mse_weight=1.0, mae_weight=1.0, NORMAL_LOSS_GLOBAL_SCALE=0.00005, LAMBDA=None):
        super().__init__()
        self.NORMAL_LOSS_GLOBAL_SCALE = NORMAL_LOSS_GLOBAL_SCALE
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self._lambda = LAMBDA

    def forward(self, input, target):
        balancing_weights = BALANCING_WEIGHTS
        weights = torch.ones_like(input) * balancing_weights[0]
        thresholds =THRESHOLDS
        for i, threshold in enumerate(thresholds):
            weights = weights + (balancing_weights[i + 1] - balancing_weights[i]) * (target >= threshold).float()
        # input: S*B*1*H*W
        # error: S*B
        mse = torch.sum(weights * ((input-target)**2), (2, 3, 4))
        mae = torch.sum(weights * (torch.abs((input-target))), (2, 3, 4))
        if self._lambda is not None:
            S, B = mse.size()
            w = torch.arange(1.0, 1.0 + S * self._lambda, self._lambda)
            if torch.cuda.is_available():
                w = w.to(mse.get_device())
            mse = (w * mse.permute(1, 0)).permute(1, 0)
            mae = (w * mae.permute(1, 0)).permute(1, 0)
        return self.NORMAL_LOSS_GLOBAL_SCALE * (self.mse_weight*torch.mean(mse) + self.mae_weight*torch.mean(mae))
    
BMSE = Weighted_mse_mae().to('cuda')
