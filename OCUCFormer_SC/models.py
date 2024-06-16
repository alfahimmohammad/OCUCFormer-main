import torch
import torch.nn as nn
from torch.autograd import Variable, grad
import numpy as np
import os 
import torch.nn.functional as F
from collections import OrderedDict
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import time
from pdb import set_trace as stx
import numbers

from einops import rearrange

from base_modules import *


class OCRN(nn.Module):
    def __init__(self,args,
                in_channels=1,
                out_channels=1, 
                dim=32,
                LayerNorm_type = 'WithBias',
                num_blocks = [1,2,3], 
                num_refinement_blocks = 3,
                heads = [1,2,4],
                ffn_expansion_factor = 2.66,
                timesteps=5,
                bias = False):
        
        super(OCRN, self).__init__()
        self.batch_size = args.batch_size
        self.dim = dim
        self.timesteps = timesteps
        self.patch_embed_in = OverlapPatchEmbed(in_channels, dim)
        self.encoder_level1 = OverlapPatchEmbed(int(dim),
                                               int(dim),
                                               LayerNorm_type=LayerNorm_type)
        self.up1_2 = Upsample(dim) ##From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim//2), 
                                                               num_heads=heads[1], 
                                                               ffn_expansion_factor=ffn_expansion_factor, 
                                                               bias=bias, 
                                                               LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        ##Residual Block
        self.residual = nn.ModuleList([OverlapPatchEmbed(int(dim//2),#int(dim//4)
                                                        int(dim//2),#int(dim//4)
                                                        LayerNorm_type=LayerNorm_type) 
                                        for _ in range(4)])
        
        self.down2_1 = Downsample(int(dim//2)) ##From Level 2 to Level 1
        self.decoder_level1 = OverlapPatchEmbed(int(dim*2**1),
                                               int(dim*2**1),
                                               LayerNorm_type=LayerNorm_type)
        self.patch_embed_out = nn.Conv2d(int(dim*2**1),out_channels, kernel_size=1, bias=bias)
        self.dc = DataConsistencyLayer(True)
        
    def forward(self,y,ksp_query_imgs,ksp_mask_query): #
        h = torch.zeros(self.batch_size,int(self.dim//2),int(y.shape[-2]*2**1),int(y.shape[-1]*2**1)).cuda()
        y = y.contiguous()

        for _ in range(self.timesteps):
            v = self.patch_embed_in(y)
            out_enc_l1 = self.encoder_level1(v)
            v = self.up1_2(out_enc_l1)
            v = self.encoder_level2(v)#out_enc_l2
            
            x = self.residual[0](h)
            h = h+x
            x = self.residual[1](h)
            h = h+x
            x = self.residual[2](h)
            h = h+x
            x = self.residual[3](h)
            h = h+x
            
            h = h+v
            
            out = self.down2_1(h)#out
            out = torch.cat([out_enc_l1,out],1)
            out = self.decoder_level1(out)
            out = self.patch_embed_out(out)
            y = self.dc(out,ksp_query_imgs,ksp_mask_query)

        return y

    
    
    
class UCRN(nn.Module):
    def __init__(self,args,
                in_channels=1,
                out_channels=1, 
                dim=32,
                LayerNorm_type = 'WithBias',
                num_blocks = [1,2,3], 
                num_refinement_blocks = 3,
                heads = [1,2,4],
                ffn_expansion_factor = 2.66,
                timesteps=5,
                bias = False):
        
        super(UCRN, self).__init__()
        self.batch_size = args.batch_size
        self.dim = dim
        self.timesteps = timesteps
        self.patch_embed_in = OverlapPatchEmbed(in_channels, dim)
        self.encoder_level1 = OverlapPatchEmbed(dim,
                                                dim,
                                                LayerNorm_type=LayerNorm_type)
        self.down1_2 = Downsample(dim) ##From Level 1 to Level 2
        self.encoder_level2 = OverlapPatchEmbed(int(dim*2**1), 
                                               int(dim*2**1),
                                               LayerNorm_type=LayerNorm_type)
        
        ##Residual Block
        self.residual = nn.ModuleList([OverlapPatchEmbed(int(dim*2**1),#int(dim*2**2)
                                                        int(dim*2**1),#int(dim*2**2)
                                                        LayerNorm_type=LayerNorm_type) 
                                        for _ in range(4)])
        self.up2_1 = Upsample(int(dim*2**1)) ##From Level 2 to Level 1
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), 
                                        num_heads=heads[0], 
                                        ffn_expansion_factor=ffn_expansion_factor, 
                                        bias=bias, 
                                        LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.patch_embed_out = nn.Conv2d(int(dim*2**1),out_channels, kernel_size=1, bias=bias)
        self.dc = DataConsistencyLayer(True)
        
    def forward(self,y,ksp_query_imgs,ksp_mask_query): #
        h = torch.zeros(self.batch_size,int(self.dim*2**1),int(y.shape[-2]//2),int(y.shape[-1]//2)).cuda()
        y = y.contiguous()

        for _ in range(self.timesteps):
            v = self.patch_embed_in(y)
            out_enc_l1 = self.encoder_level1(v)
            v = self.down1_2(out_enc_l1)
            out_enc_l2 = self.encoder_level2(v) 
            
            x = self.residual[0](h)
            h = h+x
            x = self.residual[1](h)
            h = h+x
            x = self.residual[2](h)
            h = h+x
            x = self.residual[3](h)
            h = h+x
            
            h = h+v
            
            out = self.up2_1(h)#out
            out = torch.cat([out_enc_l1,out],1)
            out = self.decoder_level1(out)
            out = self.patch_embed_out(out)
            y = self.dc(out,ksp_query_imgs,ksp_mask_query)

        return y

    

class OCUCFormer(nn.Module):
    def __init__(self,args,
                in_channels=1,
                out_channels=1,
                dim=32,
                LayerNorm_type = 'WithBias',
                heads = [1,2,4],
                num_blocks = [1,2,3], 
                num_refinement_blocks = 3,
                ffn_expansion_factor = 2.66,
                timesteps=5,
                bias = False):
        
        super(OCUCFormer, self).__init__()
        self.ocrn = OCRN(args,in_channels=in_channels,out_channels=out_channels,dim=dim,timesteps=timesteps)
        self.ucrn = UCRN(args,in_channels=in_channels,out_channels=out_channels,dim=dim,timesteps=timesteps)
        self.embed_chans_in = nn.Conv2d(out_channels,int(dim),kernel_size=1,bias=bias)
        self.refinement = nn.Sequential(*[OverlapPatchEmbed(int(dim),
                                                           int(dim),
                                                           LayerNorm_type=LayerNorm_type) for _ in range(num_refinement_blocks)])
        self.embed_chans_out = nn.Conv2d(int(dim),out_channels,kernel_size=1,bias=bias)
        self.dc = DataConsistencyLayer(True)
        
    def forward(self,x,ksp_query_imgs,ksp_mask_query): #
        
        x = self.ocrn(x,ksp_query_imgs,ksp_mask_query)
#         x = x + y
        uc_x = self.ucrn(x,ksp_query_imgs,ksp_mask_query)
        
        uc_x = self.embed_chans_in(uc_x)
        uc_x = self.refinement(uc_x)
        uc_x = self.embed_chans_out(uc_x)
        x = x + uc_x 
        x = self.dc(x,ksp_query_imgs,ksp_mask_query)
        
        return x
    
