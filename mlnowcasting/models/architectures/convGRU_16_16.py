#!/usr/bin/env python
# coding: utf-8

# functional convGRU with state transfer




import torch.nn as nn
import torch
from .layers.convGRU import ConvGRU

def _swap(x):
    return x.transpose(2, 1)
    
class Nowcasting(nn.Module):
    def __init__(self):
        
        super(Nowcasting, self).__init__()
        dtype = torch.cuda.FloatTensor

        self.conv1  = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(3,3,3), stride=(1,1,1), padding=1, dilation=1, bias=True, padding_mode='zeros')
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.02, inplace=False)
        self.rnn1   = ConvGRU(input_size=(64, 64),input_dim=16,hidden_dim=64,kernel_size=(3,3),num_layers=1,dtype=dtype,batch_first=True,bias = True,return_all_layers = False)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.02, inplace=False)
        
        self.conv2  = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3), stride=(1,2,2), padding=1, dilation=1, bias=True, padding_mode='zeros')
        self.lrelu3 = nn.LeakyReLU(negative_slope=0.02, inplace=False)
        self.rnn2   = ConvGRU(input_size=(32, 32),input_dim=64,hidden_dim=96,kernel_size=(3,3),num_layers=1,dtype=dtype,batch_first=True,bias = True,return_all_layers = False)
        self.lrelu4 = nn.LeakyReLU(negative_slope=0.02, inplace=False)
        
        self.conv3  = nn.Conv3d(in_channels=96, out_channels=96, kernel_size=(3,3,3), stride=(1,2,2), padding=1, dilation=1, bias=True, padding_mode='zeros')
        self.lrelu5 = nn.LeakyReLU(negative_slope=0.02, inplace=False)
        self.rnn3   = ConvGRU(input_size=(16, 16),input_dim=96,hidden_dim=96,kernel_size=(3,3),num_layers=1,dtype=dtype,batch_first=True,bias = True,return_all_layers = False)
        self.lrelu6 = nn.LeakyReLU(negative_slope=0.02, inplace=False)
        
        #decoder
        
        self.rnn4   = ConvGRU(input_size=(16, 16),input_dim=96,hidden_dim=96,kernel_size=(3,3),num_layers=1,dtype=dtype,batch_first=True,bias = True,return_all_layers = False)
        self.lrelu7 = nn.LeakyReLU(negative_slope=0.02, inplace=False)
        self.convt1 = nn.ConvTranspose3d(in_channels=96, out_channels=96, kernel_size=(3,4,4), stride=(1,2,2), padding=1, dilation=1, bias=True, padding_mode='zeros')
        self.lrelu8 = nn.LeakyReLU(negative_slope=0.02, inplace=False)
        
        self.rnn5    = ConvGRU(input_size=(32, 32),input_dim=96,hidden_dim=96,kernel_size=(3,3),num_layers=1,dtype=dtype,batch_first=True,bias = True,return_all_layers = False)
        self.lrelu9  = nn.LeakyReLU(negative_slope=0.02, inplace=False)
        self.convt2  = nn.ConvTranspose3d(in_channels=96, out_channels=96, kernel_size=(3,4,4), stride=(1,2,2), padding=1, dilation=1, bias=True, padding_mode='zeros')
        self.lrelu10 = nn.LeakyReLU(negative_slope=0.02, inplace=False)
        
        self.rnn6    = ConvGRU(input_size=(64, 64),input_dim=96,hidden_dim=64,kernel_size=(3,3),num_layers=1,dtype=dtype,batch_first=True,bias = True,return_all_layers = False)
        self.lrelu11 = nn.LeakyReLU(negative_slope=0.02, inplace=False)
        self.convt3  = nn.ConvTranspose3d(in_channels=64, out_channels=16, kernel_size=(3,3,3), stride=(1,1,1), padding=1, dilation=1, bias=True, padding_mode='zeros')
        self.lrelu12 = nn.LeakyReLU(negative_slope=0.02, inplace=False)
        
        self.convt4  = nn.ConvTranspose3d(in_channels=16, out_channels=1, kernel_size=(1,1,1), stride=(1,1,1), padding=0, dilation=1, bias=True, padding_mode='zeros')

    
    def forward(self, x):
        debug = 0
        if debug: print(x.shape)

        x = self.conv1(x)
        x = self.lrelu1(x)
        x = _swap(x)
        x,s1 = self.rnn1(x)
        x = self.lrelu2(x)
 
        x = _swap(x)
        x = self.conv2(x)
        x = self.lrelu3(x)
        x = _swap(x)
        x,s2 = self.rnn2(x)
        
        x = _swap(x)
        x = self.conv3(x)
        x = self.lrelu5(x)
        x = _swap(x)
        x,s3 = self.rnn3(x)
        x = self.lrelu6(x)
 
        #decoder
        x,s = self.rnn4(x, input_state=s3)
        x = self.lrelu7(x)
        x = _swap(x)   
        x = self.convt1(x)
        x = self.lrelu8(x)
        x = _swap(x)
 
        x,s = self.rnn5(x, input_state=s2)
        x = self.lrelu9(x)
        x = _swap(x)
        x = self.convt2(x)
        x = self.lrelu10(x)
        x = _swap(x)
 
        x,s = self.rnn6(x, input_state=s1)
        x = _swap(x)  
        x = self.convt3(x)
        x = self.lrelu12(x)
        x = self.convt4(x)

        return x

    
def get_model(n_filters, dropout):
     return Nowcasting()
   
