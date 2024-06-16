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
import torchvision.models as models

class DataConsistencyLayer(nn.Module):

    def __init__(self,device):

        super(DataConsistencyLayer,self).__init__()

        self.device = device

    def forward(self,predicted_img,us_kspace,us_mask):

#         us_mask_path = os.path.join(self.us_mask_path,dataset_string,mask_string,'mask_{}.npy'.format(acc_factor))

#         us_mask = torch.from_numpy(np.load(us_mask_path)).unsqueeze(2).unsqueeze(0).to(self.device)
        #print(predicted_img.shape, us_kspace.shape, us_mask.shape)
        predicted_img = predicted_img[:,0,:,:]

        #print("predicted_img: ",predicted_img.shape)
        #print("us_kspace: ", us_kspace.shape, "us_mask: ",us_mask.shape)
        kspace_predicted_img = torch.fft.fft2(predicted_img,norm = "ortho")

        #print("kspace_predicted_img: ",kspace_predicted_img.shape)
        #kspace_predicted_img_real = torch.view_as_real(kspace_predicted_img)

        us_kspace_complex = us_kspace[:,:,:,0]+us_kspace[:,:,:,1]*1j

        updated_kspace1  = us_mask * us_kspace_complex

        updated_kspace2  = (1 - us_mask) * kspace_predicted_img
        #print("updated_kspace1: ", updated_kspace1.shape, "updated_kspace2: ",updated_kspace2.shape)

        updated_kspace = updated_kspace1 + updated_kspace2
        #print("updated_kspace: ", updated_kspace.shape)
        
        #updated_kspace = updated_kspace[:,:,:,0]+updated_kspace[:,:,:,1]*1j
        #print("updated_kspace: ", updated_kspace.shape)

        updated_img  = torch.fft.ifft2(updated_kspace,norm = "ortho")
        #print("updated_img: ", updated_img.shape)

        updated_img = torch.view_as_real(updated_img)
        #print("updated_img: ", updated_img.shape)
        
        update_img_abs = updated_img[:,:,:,0] # taking real part only, change done on Sep 18 '19 bcos taking abs till bring in the distortion due to imag part also. this was verified was done by simple experiment on FFT, mask and IFFT

        update_img_abs = update_img_abs.unsqueeze(1)
        #print("updated_img_abs out of DC: ", update_img_abs.shape)

        return update_img_abs.float()



# def same_padding(images, ksizes, strides, rates):
#     assert len(images.size()) == 4
#     batch_size, channel, rows, cols = images.size()
#     out_rows = (rows + strides[0] - 1) // strides[0]
#     out_cols = (cols + strides[1] - 1) // strides[1]
#     effective_k_row = (ksizes[0] - 1) * rates[0] + 1
#     effective_k_col = (ksizes[1] - 1) * rates[1] + 1
#     padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
#     padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
#     # Pad the input
#     padding_top = int(padding_rows / 2.)
#     padding_left = int(padding_cols / 2.)
#     padding_bottom = padding_rows - padding_top
#     padding_right = padding_cols - padding_left
#     paddings = (padding_left, padding_right, padding_top, padding_bottom)
#     images = torch.nn.ZeroPad2d(paddings)(images)
#     return images, paddings


# def extract_image_patches(images, ksizes, strides, rates, padding='same'):
#     """
#     Extract patches from images and put them in the C output dimension.
#     :param padding:
#     :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
#     :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
#      each dimension of images
#     :param strides: [stride_rows, stride_cols]
#     :param rates: [dilation_rows, dilation_cols]
#     :return: A Tensor
#     """
#     assert len(images.size()) == 4
#     assert padding in ['same', 'valid']
#     paddings = (0, 0, 0, 0)

#     if padding == 'same':
#         images, paddings = same_padding(images, ksizes, strides, rates)
#     elif padding == 'valid':
#         pass
#     else:
#         raise NotImplementedError('Unsupported padding type: {}.\
#                 Only "same" or "valid" are supported.'.format(padding))

#     unfold = torch.nn.Unfold(kernel_size=ksizes,
#                              padding=0,
#                              stride=strides)
#     patches = unfold(images)
#     return patches, paddings

# class CE(nn.Module):
#     def __init__(self, ksize=7, stride_1=4, stride_2=1, softmax_scale=10,shape=64 ,p_len=64,in_channels=1,
#                  out_channels=1,inter_channels=16,use_multiple_size=False,use_topk=False,
#                  add_SE=False,num_edge = 50):
#         super(CE, self).__init__()
#         self.ksize = ksize
#         self.shape=shape
#         self.p_len=p_len
#         self.stride_1 = stride_1
#         self.stride_2 = stride_2
#         self.softmax_scale = softmax_scale
#         self.inter_channels = inter_channels
#         self.in_channels = in_channels
#         self.use_multiple_size=use_multiple_size
#         self.use_topk=use_topk
#         self.add_SE=add_SE
#         self.num_edge = num_edge
#         self.down = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=4,stride=2,
#                               padding=1)
#         self.up = nn.ConvTranspose2d(in_channels=out_channels,out_channels=out_channels,kernel_size=4,
#                                        stride=2,padding=1)
#         self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=3, stride=1,
#                            padding=1)
#         self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
#                            padding=0)
#         self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
#                                padding=0)
#         self.fc1 = nn.Sequential(
#             nn.Linear(in_features=ksize**2*inter_channels,out_features=(ksize**2*inter_channels)//4),
#             nn.ReLU()
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(in_features=ksize**2*inter_channels,out_features=(ksize**2*inter_channels)//4),
#             nn.ReLU()
#         )
#         self.thr_conv = nn.Conv2d(in_channels=in_channels,out_channels=1,kernel_size=ksize,stride=stride_1,padding=0)
#         self.bias_conv = nn.Conv2d(in_channels=in_channels,out_channels=1,kernel_size=ksize,stride=stride_1,padding=0)
#         self.out = nn.Conv2d(in_channels=self.inter_channels, out_channels=out_channels, kernel_size=3,
#                             padding=1)
#         self.last = nn.Conv2d(in_channels=2*in_channels,out_channels=out_channels, kernel_size=3, padding=1)
        
#     def forward(self, b):
#         b_1 = self.down(b)
#         b1 = self.g(b_1)
#         b2 = self.theta(b_1)
#         b3 = b1

#         raw_int_bs = list(b1.size())  # b*c*h*w
#         b4, _ = same_padding(b_1,[self.ksize,self.ksize],[self.stride_1,self.stride_1],[1,1])
#         soft_thr = self.thr_conv(b4).view(raw_int_bs[0],-1)
#         soft_bias = self.bias_conv(b4).view(raw_int_bs[0],-1)
#         patch_28, paddings_28 = extract_image_patches(b1, ksizes=[self.ksize, self.ksize],
#                                                       strides=[self.stride_1, self.stride_1],
#                                                       rates=[1, 1],
#                                                       padding='same')
        
#         patch_28 = patch_28.view(raw_int_bs[0], raw_int_bs[1], self.ksize, self.ksize, -1)
#         patch_28 = patch_28.permute(0, 4, 1, 2, 3)
#         patch_28_group = torch.split(patch_28, 1, dim=0)

#         patch_112, paddings_112 = extract_image_patches(b2, ksizes=[self.ksize, self.ksize],
#                                                         strides=[self.stride_2, self.stride_2],
#                                                         rates=[1, 1],
#                                                         padding='same')
        
#         patch_112 = patch_112.view(raw_int_bs[0], raw_int_bs[1], self.ksize, self.ksize, -1)
#         patch_112 = patch_112.permute(0, 4, 1, 2, 3)
#         patch_112_group = torch.split(patch_112, 1, dim=0)

#         patch_112_2, paddings_112_2 = extract_image_patches(b3, ksizes=[self.ksize, self.ksize],
#                                                         strides=[self.stride_2, self.stride_2],
#                                                         rates=[1, 1],
#                                                         padding='same')
        
#         patch_112_2 = patch_112_2.view(raw_int_bs[0], raw_int_bs[1], self.ksize, self.ksize, -1)
#         patch_112_2 = patch_112_2.permute(0, 4, 1, 2, 3)
#         patch_112_group_2 = torch.split(patch_112_2, 1, dim=0)
#         y = []
#         w, h = raw_int_bs[2], raw_int_bs[3]
#         _, paddings = same_padding(b3[0,0].unsqueeze(0).unsqueeze(0), [self.ksize, self.ksize], [self.stride_2, self.stride_2], [1, 1])
#         itr = 0
#         for xi, wi,pi,thr,bias in zip(patch_112_group_2, patch_28_group, patch_112_group,soft_thr,soft_bias):
#             c_s = pi.shape[2]
#             k_s = wi[0].shape[2]
#             wi = self.fc1(wi.view(wi.shape[1],-1))
#             xi = self.fc2(xi.view(xi.shape[1],-1)).permute(1,0)
#             score_map = torch.matmul(wi,xi)
#             score_map = score_map.view(1, score_map.shape[0], math.ceil(w / self.stride_2),
#                                        math.ceil(h / self.stride_2))
#             b_s, l_s, h_s, w_s = score_map.shape
#             yi = score_map.view(l_s, -1)
            
#             mask = F.relu(yi-yi.mean(dim=1,keepdim=True)*thr.unsqueeze(1)+bias.unsqueeze(1))
#             mask_b = (mask!=0.).float()

#             yi = yi * mask
#             yi = F.softmax(yi * self.softmax_scale, dim=1)
#             yi = yi * mask_b
            
#             pi = pi.view(h_s * w_s, -1)
#             yi = torch.mm(yi, pi)
#             yi = yi.view(b_s, l_s, c_s, k_s, k_s)[0]
#             zi = yi.view(1, l_s, -1).permute(0, 2, 1)
#             zi = torch.nn.functional.fold(zi, (raw_int_bs[2], raw_int_bs[3]), (self.ksize, self.ksize), padding=paddings[0], stride=self.stride_1)
#             inp = torch.ones_like(zi)
#             inp_unf = torch.nn.functional.unfold(inp, (self.ksize, self.ksize), padding=paddings[0], stride=self.stride_1)
#             out_mask = torch.nn.functional.fold(inp_unf, (raw_int_bs[2], raw_int_bs[3]), (self.ksize, self.ksize), padding=paddings[0], stride=self.stride_1)
#             out_mask += (out_mask==0.).float()
#             zi = zi / out_mask
#             y.append(zi)
#         y = torch.cat(y, dim=0)
#         y = self.out(y)
#         y = self.up(y)
#         z = torch.cat((b,y),dim=1)
#         z = self.last(z)
#         return z
    
class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'


class UnetModel(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2
        self.conv = ConvBlock(ch, ch, drop_prob)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, drop_prob)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, drop_prob)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = input
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)

        output = self.conv(output)
        output_latent = output
        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat([output, stack.pop()], dim=1)
            output = layer(output)
        return self.conv2(output)
    

class five_layerCNN_MAML(nn.Module):
    def __init__(self,args,n_ch=1):

        super(five_layerCNN_MAML,self).__init__()

        self.relu = nn.ReLU()
        
        self.BaseCNN = nn.Sequential(OrderedDict([
            ('conv1',nn.Conv2d(n_ch,32,3,1,1,bias=False)),
            ('bn1',nn.BatchNorm2d(32)),
            ('relu1', nn.ReLU()),
            ('conv2',nn.Conv2d(32,32,3,1,1,bias=False)),
            ('bn2',nn.BatchNorm2d(32)),
            ('relu2', nn.ReLU()),
            ('conv3',nn.Conv2d(32,32,3,1,1,bias=False)),
            ('bn3',nn.BatchNorm2d(32)),
            ('relu3', nn.ReLU()),
            ('conv4',nn.Conv2d(32,32,3,1,1,bias=False)),
            ('bn4',nn.BatchNorm2d(32)),
            ('relu4', nn.ReLU()),
            ('conv5',nn.Conv2d(32,n_ch,3,1,1,bias=False)),
        ]))
        
    def forward(self,us_query_input):
        
        cnn_query_output = self.BaseCNN(us_query_input)+us_query_input
        
        return cnn_query_output

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False, LayerNorm_type='BiasFree'):
        super(OverlapPatchEmbed, self).__init__()

        self.proj_dw = nn.Conv2d(in_c, in_c, kernel_size=3, stride=1, padding=1, groups=in_c, bias=bias)
        self.proj_pw = nn.Conv2d(in_c, embed_dim, kernel_size=1, stride=1, padding=0, bias=bias)
        self.norm = LayerNorm(embed_dim, LayerNorm_type)
        

    def forward(self, x, Norm=False):
        x = self.proj_dw(x)
        x = self.proj_pw(x)
        if Norm:
            x = self.norm(x) #Added Norm only for UCOCRestormNetLite

        return F.leaky_relu(x) #Added leaky relu for every other model except CNN_Restormer



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, groups=n_feat, bias=False),
                                  nn.Conv2d(n_feat, n_feat//2, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, groups=n_feat, bias=False),
                                  nn.Conv2d(n_feat, n_feat*2, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################