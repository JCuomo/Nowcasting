# not convex
import torch

def get_loss_fx():
    return metricLoss


def metricLoss(target, prediction):
    device = target.device
    th = 0.8
    one = torch.tensor(1.0).to(device)
    zero = torch.tensor(0.0).to(device)

    target = torch.where(target > th, one, zero)
    prediction = torch.where(prediction > th, one, zero)
    
    combined = target*2 + prediction # factor 2 is to distinguish the contribution in the SUM from each term
    hits = (combined==3).sum().float()
    misses = (combined==2).sum().float()
    false_alarms = (combined==1).sum().float()
    Rainfall_MSE = torch.mean((target - prediction)**2)
    CSI = (hits/(hits+misses+false_alarms))
    FAR = (false_alarms/(hits+false_alarms))
    POD = (hits/(hits+misses))
    loss =  FAR #+ Rainfall_MSE
    return torch.autograd.Variable(loss, requires_grad = True)  
    
