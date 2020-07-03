#!/usr/bin/env python
# coding: utf-8

import torch.nn as nn
import torch

from .layers.trajGRU import TrajGRU


    
def get_model(n_filters=128, dropout=1):
     return Nowcasting()

class Nowcasting(nn.Module):
    def __init__(self):
        
        super(Nowcasting, self).__init__()

        self.conv1  = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(3,3,3), stride=(1,1,1), padding=1, dilation=1, bias=True, padding_mode='zeros')
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.02, inplace=False)
        self.rnn1   = TrajGRU(input_channel=16, num_filter=64, b_h_w=(4, 64, 64), zoneout=0.0, L=13,
                                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                                h2h_kernel=(5, 5), h2h_dilate=(1, 1))
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.02, inplace=False)
        
        self.conv2  = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3), stride=(1,2,2), padding=1, dilation=1, bias=True, padding_mode='zeros')
        self.lrelu3 = nn.LeakyReLU(negative_slope=0.02, inplace=False)
        self.rnn2   = TrajGRU(input_channel=64, num_filter=96, b_h_w=(4, 32, 32), zoneout=0.0, L=13,
                                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                                h2h_kernel=(5, 5), h2h_dilate=(1, 1))
        self.lrelu4 = nn.LeakyReLU(negative_slope=0.02, inplace=False)
        
        self.conv3  = nn.Conv3d(in_channels=96, out_channels=96, kernel_size=(3,3,3), stride=(1,2,2), padding=1, dilation=1, bias=True, padding_mode='zeros')
        self.lrelu5 = nn.LeakyReLU(negative_slope=0.02, inplace=False)
        self.rnn3   = TrajGRU(input_channel=96, num_filter=96, b_h_w=(4, 16, 16), zoneout=0.0, L=13,
                                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                                h2h_kernel=(5, 5), h2h_dilate=(1, 1))
        self.lrelu6 = nn.LeakyReLU(negative_slope=0.02, inplace=False)
        
        #decoder
        
        self.rnn4   = TrajGRU(input_channel=96, num_filter=96, b_h_w=(4, 16, 16), zoneout=0.0, L=13,
                                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                                h2h_kernel=(5, 5), h2h_dilate=(1, 1))
        self.lrelu7 = nn.LeakyReLU(negative_slope=0.02, inplace=False)
        self.convt1 = nn.ConvTranspose3d(in_channels=96, out_channels=96, kernel_size=(3,4,4), stride=(1,2,2), padding=1, dilation=1, bias=True, padding_mode='zeros')
        self.lrelu8 = nn.LeakyReLU(negative_slope=0.02, inplace=False)
        
        self.rnn5   = TrajGRU(input_channel=96, num_filter=96, b_h_w=(4, 32, 32), zoneout=0.0, L=13,
                                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                                h2h_kernel=(5, 5), h2h_dilate=(1, 1))
        self.lrelu9  = nn.LeakyReLU(negative_slope=0.02, inplace=False)
        self.convt2  = nn.ConvTranspose3d(in_channels=96, out_channels=96, kernel_size=(3,4,4), stride=(1,2,2), padding=1, dilation=1, bias=True, padding_mode='zeros')
        self.lrelu10 = nn.LeakyReLU(negative_slope=0.02, inplace=False)
        
        self.rnn6   = TrajGRU(input_channel=96, num_filter=64, b_h_w=(4, 64, 64), zoneout=0.0, L=13,
                                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                                h2h_kernel=(5, 5), h2h_dilate=(1, 1))
        self.lrelu11 = nn.LeakyReLU(negative_slope=0.02, inplace=False)
        self.convt3  = nn.ConvTranspose3d(in_channels=64, out_channels=16, kernel_size=(3,3,3), stride=(1,1,1), padding=1, dilation=1, bias=True, padding_mode='zeros')
        self.lrelu12 = nn.LeakyReLU(negative_slope=0.02, inplace=False)
        
        self.convt4  = nn.ConvTranspose3d(in_channels=16, out_channels=1, kernel_size=(1,1,1), stride=(1,1,1), padding=0, dilation=1, bias=True, padding_mode='zeros')

    
    def forward(self, x):
        debug = 0
        if debug: print(x.shape)

        x = self.conv1(x)
        x = self.lrelu1(x)
        x,s1 = self.rnn1(x, seq_len=16)
        x = self.lrelu2(x)
 
        if debug: print(x.shape)
        x = self.conv2(x)
        x = self.lrelu3(x)
        x,s2 = self.rnn2(x, seq_len=16)
        
        if debug: print(x.shape)
        x = self.conv3(x)
        x = self.lrelu5(x)
        x,s3 = self.rnn3(x, seq_len=16)
        x = self.lrelu6(x)
 
        if debug: print(x.shape)
        #decoder
        x,s = self.rnn4(x, seq_len=16)
        x = self.lrelu7(x)
        x = self.convt1(x)
        x = self.lrelu8(x)
 
        if debug: print(x.shape)
        x,s = self.rnn5(x, seq_len=16)
        x = self.lrelu9(x)
        x = self.convt2(x)
        x = self.lrelu10(x)
 
        if debug: print(x.shape)
        x,s = self.rnn6(x, seq_len=16)
        x = self.convt3(x)
        x = self.lrelu12(x)
        x = self.convt4(x)

        if debug: print(x.shape)
        return x





