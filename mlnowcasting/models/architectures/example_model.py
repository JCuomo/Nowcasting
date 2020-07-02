#!/usr/bin/env python
# coding: utf-8


import torch.nn as nn
import torch



def downsampling(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels,  out_channels, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.02, inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.02, inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.02, inplace=True),
        nn.BatchNorm3d(out_channels, affine=True),
        nn.Dropout(0.2)
    ) 
def upsampling(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels,  out_channels, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.02, inplace=True),
        nn.ConvTranspose3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.02, inplace=True),
        nn.ConvTranspose3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.02, inplace=True),
        nn.BatchNorm3d(out_channels, affine=True),
        nn.Dropout(0.2)
    ) 
def upsamplingLast(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels,  out_channels, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.02, inplace=True),
        nn.ConvTranspose3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.02, inplace=True),
        nn.ConvTranspose3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.Sigmoid()
    ) 

class Nowcasting(nn.Module):
    def __init__(self, n_filter, dropout):
        
        super(Nowcasting, self).__init__()
        dtype = torch.cuda.FloatTensor

        self.e_conv0 = downsampling(in_channels=1,          out_channels=n_filter)
        self.e_conv1 = downsampling(in_channels=n_filter,   out_channels=n_filter*2)
        self.e_conv2 = downsampling(in_channels=n_filter*2, out_channels=n_filter*3)
        self.e_conv3 = downsampling(in_channels=n_filter*3, out_channels=n_filter*4)
        
        self.d_conv3 = upsampling(in_channels=n_filter*4, out_channels=n_filter*3)
        self.d_conv2 = upsampling(in_channels=n_filter*3, out_channels=n_filter*2)
        self.d_conv1 = upsampling(in_channels=n_filter*2, out_channels=n_filter)
        self.d_conv0 = upsamplingLast(in_channels=n_filter,   out_channels=1)
        
    def forward(self, x):
        e1 = self.e_conv0(x)
        e2 = self.e_conv1(e1)
        e3 = self.e_conv2(e2)
        e4 = self.e_conv3(e3)
        
        d3 = self.d_conv3(e4)
        d2 = self.d_conv2(d3+e3)
        d1 = self.d_conv1(d2+e2)
        y = self.d_conv0(d1+e1)

        return y

    
def get_model(n_filters, dropout):
     return Nowcasting(n_filters, dropout)
    

