import torch
import torch.nn as nn
from data import transforms as T
import functools
import sys
from models import *
from base_modules import *

class dataConsistencyTerm(nn.Module):

    def __init__(self, noise_lvl=None):
        super(dataConsistencyTerm, self).__init__()
        self.noise_lvl = noise_lvl
        if noise_lvl is not None:
            self.noise_lvl = torch.nn.Parameter(torch.Tensor([noise_lvl]))

    def perform(self, x, k0, mask, sensitivity):

        """
        k    - input in k-space
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """
        #print('1:',x.shape,sensitivity.shape)
        x = T.complex_multiply(x[...,0].unsqueeze(1), x[...,1].unsqueeze(1), 
                               sensitivity[...,0], sensitivity[...,1])
     
        #print('2:',x.shape)
        #sys.exit()
        #k = torch.fft.fft(x, 2, norm='ortho')
        k = T.dc_fft2(x)
              
        v = self.noise_lvl
        if v is not None: # noisy case
            # out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
            out = (1 - mask) * k + mask * (v * k + (1 - v) * k0) 
        else:  # noiseless case
            out = (1 - mask) * k + mask * k0
    
        # ### backward op ### #
        #x = torch.fft.ifft(out, 2, norm='ortho')
        x = T.dc_ifft2(out)
        #print("x: ",x.shape, "out: ", out.shape, "Sens: ", sensitivity.shape)      
        Sx = T.complex_multiply(x[...,0], x[...,1], 
                                sensitivity[...,0], 
                               -sensitivity[...,1]).sum(dim=1)     
        
        SS = T.complex_multiply(sensitivity[...,0], 
                                sensitivity[...,1], 
                                sensitivity[...,0], 
                               -sensitivity[...,1]).sum(dim=1)
        #print("Sx: ", Sx.shape) 
        return Sx, SS

    
class weightedAverageTerm(nn.Module):

    def __init__(self, para=None):
        super(weightedAverageTerm, self).__init__()
        self.para = para
        if para is not None:
            self.para = torch.nn.Parameter(torch.Tensor([para]))

    def perform(self, cnn, Sx, SS):
        
        x = self.para*cnn + (1 - self.para)*Sx
#        print("sx: ", Sx.shape, "cnn: ", cnn.shape, "x: ", x.shape)
        return x



class cnn_layer(nn.Module):
    
    def __init__(self):
        super(cnn_layer, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(2,  64, 3, padding=1, bias=True),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2,  3, padding=1, bias=True)
        )  
#         self.initialize_weights()
        
#     def initialize_weights(self):
#         for item in self.conv:
#             if isinstance(item, nn.Conv2d):
#                 torch.nn.init.kaiming_uniform_(item.weight)
#             elif isinstance(item, nn.InstanceNorm2d):
#                 torch.nn.init.normal_(item.weight)
        
        
    def forward(self, x):
        
        #print(x.shape)
        x = x.permute(0, 3, 1, 2)
        #print('a:',x.shape)
        x = self.conv(x)
        #print('b:',x.shape)
        x = x.permute(0, 2, 3, 1)
        #print('c:',x.shape)

        return x
    

class ocucformer_network(nn.Module):
    
    def __init__(self, args, alfa=1, beta=1, cascades=1):
        super(ocucformer_network, self).__init__()
        
        self.cascades = cascades 
        conv_blocks = []
        dc_blocks = []
        wa_blocks = []
        
        for i in range(cascades):
            conv_blocks.append(OCUCFormer(args)) 
            dc_blocks.append(dataConsistencyTerm(alfa)) 
            wa_blocks.append(weightedAverageTerm(beta)) 
        
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dc_blocks = nn.ModuleList(dc_blocks)
        self.wa_blocks = nn.ModuleList(wa_blocks)
        
        print(self.conv_blocks)
        print(self.dc_blocks)
        print(self.wa_blocks)
 
    def forward(self, x, k, m, c):
        shape = x.shape
                
        for i in range(self.cascades):
            x_crop = T.complex_center_crop(x,(320,320))
            x_crop = x_crop.permute(0, 3, 1, 2)
            x_cnn_crop = self.conv_blocks[i](x_crop,k,m,c,shape)
            x_cnn_crop = x_cnn_crop.permute(0, 2, 3, 1)
            x_cnn_pad = T.complex_img_pad(x_cnn_crop,shape)
            Sx, SS = self.dc_blocks[i].perform(x_cnn_pad, k, m, c)
            x = self.wa_blocks[i].perform(x + x_cnn_pad, Sx, SS)
    
        x = T.complex_center_crop(x,(320,320))
        return x   
