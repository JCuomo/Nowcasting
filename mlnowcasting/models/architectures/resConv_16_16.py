#!/usr/bin/env python
# coding: utf-8

# skip connection between layers of same dim
import torch.nn as nn
import torch


def downsampling(in_channels, out_channels, typ='all'):
    if typ == 'res':
        k=(3,4,4)
        s=(1,2,2)
    elif typ == 'seq':
        k=(4,3,3)
        s=(2,1,1)
    else:
        k=(4,4,4)
        s=(2,2,2)
        
    return nn.Sequential(
        nn.Conv3d(in_channels,  out_channels, kernel_size=k, stride=s, padding=1),
        nn.LeakyReLU(negative_slope=0.02, inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.02, inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.02, inplace=True),
        nn.BatchNorm3d(out_channels, affine=True)
    ) 
 
def upsampling(in_channels, out_channels, typ='all'):
    if typ == 'res':
        k=(3,4,4)
        s=(1,2,2)
    elif typ == 'seq':
        k=(4,3,3)
        s=(2,1,1)
    else:
        k=(4,4,4)
        s=(2,2,2)
        
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels,  out_channels, kernel_size=k, stride=s, padding=1),
        nn.LeakyReLU(negative_slope=0.02, inplace=True),
        nn.ConvTranspose3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.02, inplace=True),
        nn.ConvTranspose3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.02, inplace=True),
        nn.BatchNorm3d(out_channels, affine=True)
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
        debug = 0
        
        if debug: print(x.shape)
        e1 = self.e_conv0(x)
        if debug: print(e1.shape)
        e2 = self.e_conv1(e1)
        if debug: print(e2.shape)
        e3 = self.e_conv2(e2)
        if debug: print(e3.shape)
        e4 = self.e_conv3(e3)
        if debug: print(e4.shape)
        
        d3 = self.d_conv3(e4)
        if debug: print(d3.shape)
        d2 = self.d_conv2(d3+e3)
        if debug: print(d2.shape)
        d1 = self.d_conv1(d2+e2)
        if debug: print(d1.shape)
        y = self.d_conv0(d1+e1)
        if debug: print(y.shape)
        if debug: print('----------------')
        
        #y = self.f_conv0(d0)
        return y


def get_model(n_filters, dropout):
     return Nowcasting(n_filters, dropout)




