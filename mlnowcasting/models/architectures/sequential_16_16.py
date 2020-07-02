#!/usr/bin/env python
# coding: utf-8

import torch.nn as nn
import torch


def get_model(n_filter = 128, dropout=0):
    
    model = nn.Sequential(
        nn.Conv3d(in_channels=1, out_channels=n_filter, kernel_size=(4,4,4), stride=(2,2,2), padding=1, dilation=1, bias=True, padding_mode='zeros'),

        nn.Conv3d(in_channels=n_filter, out_channels=n_filter*2, kernel_size=(4,4,4), stride=(2,2,2), padding=1, dilation=1, bias=True, padding_mode='zeros'),   
        nn.BatchNorm3d(n_filter*2, affine=False),
        nn.LeakyReLU(negative_slope=0.02, inplace=False),

        nn.Conv3d(in_channels=n_filter*2, out_channels=n_filter*3, kernel_size=(4,4,4), stride=(2,2,2), padding=1, dilation=1, bias=True, padding_mode='zeros'),   
        nn.BatchNorm3d(n_filter*3, affine=False),
        nn.LeakyReLU(negative_slope=0.02, inplace=False),    

        nn.Conv3d(in_channels=n_filter*3, out_channels=n_filter*4, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), dilation=1, bias=True, padding_mode='zeros'),   
        nn.BatchNorm3d(n_filter*4, affine=False),
        nn.LeakyReLU(negative_slope=0.02, inplace=False),

        nn.ConvTranspose3d(in_channels=n_filter*4, out_channels=n_filter*8, kernel_size=4, stride=2, padding=1, dilation=1, bias=True, padding_mode='zeros'),

        nn.ConvTranspose3d(in_channels=n_filter*8, out_channels=n_filter*4, kernel_size=(4,4,4), stride=(2,2,2), padding=1, dilation=1, bias=True, padding_mode='zeros'),
        nn.BatchNorm3d(n_filter*4, affine=False),
        nn.LeakyReLU(negative_slope=0.02, inplace=False),

        nn.ConvTranspose3d(in_channels=n_filter*4, out_channels=n_filter*2, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), dilation=1, bias=True, padding_mode='zeros'),
        nn.BatchNorm3d(n_filter*2, affine=False),
        nn.LeakyReLU(negative_slope=0.02, inplace=False),

        nn.ConvTranspose3d(in_channels=n_filter*2, out_channels=n_filter, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1), dilation=1, bias=True, padding_mode='zeros'),
        nn.BatchNorm3d(n_filter, affine=False),
        nn.LeakyReLU(negative_slope=0.02, inplace=False),

        nn.ConvTranspose3d(in_channels=n_filter, out_channels=1, kernel_size=(3,3,3), stride=(1,1,1), padding=1, dilation=1, bias=True, padding_mode='zeros')
    )
    return model

