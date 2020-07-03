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
         # detect if CUDA is available or not
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            dtype = torch.cuda.FloatTensor # computation in GPU
        else:
            dtype = torch.FloatTensor

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
        self.e_rnn0  =   RNN(H_W/1, in_channels=1, out_channels=1)
        
        self.e_down1 = downsampling(in_channels=1, out_channels=n_filter)       
        self.e_rnn1  =   RNN(H_W/2, in_channels=n_filter*1, out_channels=n_filter*1)
        
        self.e_down2 = downsampling(in_channels=n_filter*1, out_channels=n_filter*2)
        self.e_rnn2  =   RNN(H_W/4, in_channels=n_filter*2, out_channels=n_filter*2)
        
        self.e_down3 = downsampling(in_channels=n_filter*2, out_channels=n_filter*3)
        self.e_rnn3  =   RNN(H_W/8, in_channels=n_filter*3, out_channels=n_filter*3)
        #self.e_down4 = downsampling(in_channels=n_filter*3, out_channels=n_filter*4)
        
        self.d_up3   = nn.ConvTranspose3d(n_filter*3, n_filter*3, kernel_size=3, stride=1, padding=1)
        #self.d_up4  = upsampling(in_channels=n_filter*3, out_channels=n_filter*3)
        self.d_rnn3 = RNN(H_W/8, in_channels=n_filter*3, out_channels=n_filter*3)

        self.d_up2  = upsampling(in_channels=n_filter*3, out_channels=n_filter*2)
        self.d_rnn2 = RNN(H_W/4, in_channels=n_filter*2, out_channels=n_filter*2)

        self.d_up1  = upsampling(in_channels=n_filter*2, out_channels=n_filter*1)
        self.d_rnn1 = RNN(H_W/2, in_channels=n_filter*1, out_channels=n_filter*1)

        self.d_up0  = upsampling(in_channels=n_filter*1, out_channels=1)
        self.d_rnn0 = RNN(H_W/1, in_channels=1, out_channels=1)
        
        self.d_conv0 = nn.ConvTranspose3d(in_channels=1,  out_channels=1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        
        e0_con    = self.e_conv0(x);                                                  print('e0_con',e0_con.shape)
        e0_rnn,s0 = self.e_rnn0(e0_con);                                              print('e0_rnn',e0_rnn.shape)
        e1_con    = self.e_down1(e0_rnn);                                             print(' e1_con',e1_con.shape)
        e1_rnn,s1 = self.e_rnn1(e1_con);                                              print(' e1_rnn',e1_rnn.shape)
        e2_con    = self.e_down2(e1_rnn);                                             print('   e2_con',e2_con.shape)
        e2_rnn,s2 = self.e_rnn2(e2_con);                                              print('   e2_rnn',e2_rnn.shape)
        e3_con    = self.e_down3(e2_rnn);                                             print('     e3_con',e3_con.shape)
        e3_rnn,s3 = self.e_rnn3(e3_con);                                              print('     e3_rnn',e3_rnn.shape)
        #e4 = self.e_down4(e4_rnn);                                                   print('       e4_con',e4.shape)
        ##
        d3_con    = self.d_up3(e3_rnn);                                               print('     d3_con',d3_con.shape)
        d3_rnn,_  = self.d_rnn3(d3_con+e3_con, hidden_state=None, input_state=s3);    print('     d3_rnn',d3_rnn.shape)
        d2_con    = self.d_up2(d3_rnn);                                               print('   d2_con',d2_con.shape)
        d2_rnn,_  = self.d_rnn2(d2_con+e2_con, hidden_state=None, input_state=s2);    print('   d2_rnn',d2_rnn.shape)
        d1_con    = self.d_up1(d2_rnn);                                               print(' d1_con',d1_con.shape)
        d1_rnn,_  = self.d_rnn1(d1_con+e1_con, hidden_state=None, input_state=s1);    print(' d1_rnn',d1_rnn.shape)
        d0_con    = self.d_up0(d1_rnn);                                               print('d0_con',d0_con.shape)
        d0_rnn,_  = self.d_rnn0(d0_con+e0_con, hidden_state=None, input_state=s0);    print('d0_rnn',d0_rnn.shape)
        #
        y = self.d_conv0(d0_rnn);                                                     print('y', y.shape)
        
        return self.sigmoid(y)

    
def get_model(n_filters, dropout):
     return Nowcasting(n_filters, dropout)
   
