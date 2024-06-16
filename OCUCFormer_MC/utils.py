import torch
import numpy as np
import sys


def npComplexToTorch(kspace_np):

    # Converts a numpy complex to torch 
    kspace_real_torch=torch.from_numpy(kspace_np.real)
    kspace_imag_torch=torch.from_numpy(kspace_np.imag)
    kspace_torch = torch.stack([kspace_real_torch,kspace_imag_torch],dim=2)
    
    return kspace_torch

def gradient_penalty(critic, real, fake, device):
    real = real.permute(0,3,1,2)
    fake = fake.permute(0,3,1,2)
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)
    interpolated_images = interpolated_images.permute(0,2,3,1)
    # Calculate critic scores
    mixed_scores = critic(interpolated_images)
    #print(mixed_scores.shape, interpolated_images.shape)

    gradient = torch.autograd.grad(inputs=interpolated_images,
                                   outputs=mixed_scores,
                                   grad_outputs=torch.ones_like(mixed_scores),
                                   create_graph=True,
                                   retain_graph=True)[0]
    #print(gradient.shape)
    #sys.exit(0)
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty