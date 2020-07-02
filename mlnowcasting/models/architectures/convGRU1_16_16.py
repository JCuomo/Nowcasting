#!/usr/bin/env python
# coding: utf-8

# functional convGRU with state transfer




import torch.nn as nn
import torch
from .layers.convGRU import ConvGRU
import numpy as np

class RNN(nn.Module):
    def swap(self, x):
        return x.transpose(2, 1)
    
    def __init__(self, H_W, in_channels, out_channels, k=(3,3), n_layers=1):
        
        super(RNN, self).__init__()
        dtype = torch.cuda.FloatTensor

        self.rnn = ConvGRU(input_size = tuple([int(x) for x in H_W]),
                           input_dim = in_channels,
                           hidden_dim = out_channels,
                           kernel_size = k,
                           num_layers = n_layers,
                           dtype = dtype,
                           batch_first = True,
                           bias = True,
                           return_all_layers = False)
        
    def forward(self, x, hidden_state=None, input_state=None):
        x1 = self.swap(x)
        x2, state = self.rnn(x1, hidden_state, input_state)
        x3 = self.swap(x2)
        return x3, state


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

class Nowcasting(nn.Module):
    def __init__(self, n_filter, dropout):
        
        super(Nowcasting, self).__init__()
        dtype = torch.cuda.FloatTensor
        H_W = np.array([64,64], dtype='uint8')
        self.e_conv0   = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode='zeros')
        #self.act0    = nn.LeakyReLU(negative_slope=0.02, inplace=False)
        
        self.e_rnn1  =   RNN(H_W/1, in_channels=1, out_channels=n_filter)
        self.e_down1 = downsampling(in_channels=n_filter, out_channels=n_filter)
        
        self.e_rnn2  =   RNN(H_W/2, in_channels=n_filter*1, out_channels=n_filter*2)
        self.e_down2 = downsampling(in_channels=n_filter*2, out_channels=n_filter*2)
                
        self.e_rnn3  =   RNN(H_W/4, in_channels=n_filter*2, out_channels=n_filter*3)
        self.e_down3 = downsampling(in_channels=n_filter*3, out_channels=n_filter*3)
        
        self.e_rnn4  =   RNN(H_W/8, in_channels=n_filter*3, out_channels=n_filter*4)
        self.e_down4 = downsampling(in_channels=n_filter*4, out_channels=n_filter*4)
        
        
        self.d_up4  = upsampling(in_channels=n_filter*4, out_channels=n_filter*4)
        self.d_rnn4 = RNN(H_W/8, in_channels=n_filter*4, out_channels=n_filter*4)

        self.d_up3  = upsampling(in_channels=n_filter*4, out_channels=n_filter*4)
        self.d_rnn3 = RNN(H_W/4, in_channels=n_filter*4, out_channels=n_filter*3)

        self.d_up2  = upsampling(in_channels=n_filter*3, out_channels=n_filter*3)
        self.d_rnn2 = RNN(H_W/2, in_channels=n_filter*3, out_channels=n_filter*2)

        self.d_up1  = upsampling(in_channels=n_filter*2, out_channels=n_filter*2)
        self.d_rnn1 = RNN(H_W/1, in_channels=n_filter*2, out_channels=n_filter*1)
        
        self.d_conv0 = nn.ConvTranspose3d(in_channels=n_filter,  out_channels=1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        
        e0 = self.e_conv0(x);                                             #print('e0',e0.shape)
        e1_rnn,s1 = self.e_rnn1(e0);                                      #print('e1_rnn',e1_rnn.shape)
        e1 = self.e_down1(e1_rnn);                                        #print('e1',e1.shape)
        e2_rnn,s2 = self.e_rnn2(e1);                                      #print('e2_rnn',e2_rnn.shape)
        e2 = self.e_down2(e2_rnn);                                        #print('e2',e2.shape)
        e3_rnn,s3 = self.e_rnn3(e2);                                      #print('e3_rnn',e3_rnn.shape)
        e3 = self.e_down3(e3_rnn);                                        #print('e3',e3.shape)
        e4_rnn,s4 = self.e_rnn4(e3);                                      #print('e4_rnn',e4_rnn.shape)
        e4 = self.e_down4(e4_rnn);                                        #print('e4',e4.shape)
        
        d4 = self.d_up4(e4);                                             #print('d4',d4.shape)
        d4_rnn,_ = self.d_rnn4(d4, hidden_state=None, input_state=s4);   #print('d4_rnn',d4_rnn.shape)
        d3 = self.d_up3(d4_rnn);                                         #print('d3',d3.shape)
        d3_rnn,_ = self.d_rnn3(d3, hidden_state=None, input_state=s3);   #print('d3_rnn',d3_rnn.shape)
        d2 = self.d_up2(d3_rnn);                                         #print('d2',d2.shape)
        d2_rnn,_ = self.d_rnn2(d2, hidden_state=None, input_state=s2);   #print('d2_rnn',d2_rnn.shape)
        d1 = self.d_up1(d2_rnn);                                         #print('d1',d1.shape)
        d1_rnn,_ = self.d_rnn1(d1, hidden_state=None, input_state=s1);   #print('d1_rnn',d1_rnn.shape)
        y = self.d_conv0(d1_rnn);                                        #print('y', y.shape)
        
        return self.sigmoid(y)

    
def get_model(n_filters, dropout):
     return Nowcasting(n_filters, dropout)
   
