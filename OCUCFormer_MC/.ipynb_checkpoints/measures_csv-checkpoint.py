import argparse
import pathlib
from argparse import ArgumentParser

import h5py
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
import pandas as pd 
from data import transforms as T
import torch
from torch.nn import functional as F

def complex_img_pad(im_crop, shape):

    _, h1, w1, _ = im_crop.shape 
    h2, w2 = shape

    
    h_dif = h2 - h1 
    w_dif = w2 - w1

    # how this will work for odd data

    h_dif_half = h_dif // 2
    w_dif_half = w_dif // 2

    im_crop_pad = F.pad(im_crop,[0,0,w_dif_half, w_dif_half, h_dif_half, h_dif_half])

    return im_crop_pad


def mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)


def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())


def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return structural_similarity(gt,pred,data_range=gt.max())


def evaluate(args, recons_key,metrics_info):

    for tgt_file in tqdm(args.target_path.iterdir()):
        with h5py.File(tgt_file) as target, h5py.File(args.predictions_path / tgt_file.name) as recons:
            sensitivity = torch.from_numpy(target['sensitivity'][:])#F
            target = torch.from_numpy(target[recons_key][:])
            target = T.complex_abs(T.complex_center_crop(target,(320,320))).numpy()
            recons = recons['reconstruction'][:]
            recons = np.transpose(recons,[1,2,0])
            if len(target.shape) ==2 :
                target = np.expand_dims(target,2)

            no_slices = target.shape[-1]

            for index in range(no_slices):
                target_slice = target[:,:,index]
                recons_slice = recons[:,:,index]
                mse_slice  = round(mse(target_slice,recons_slice),5)
                nmse_slice = round(nmse(target_slice,recons_slice),5)
                psnr_slice = round(psnr(target_slice,recons_slice),2)
                ssim_slice = round(ssim(target_slice,recons_slice),4)

                metrics_info['MSE'].append(mse_slice)
                metrics_info['NMSE'].append(nmse_slice)
                metrics_info['PSNR'].append(psnr_slice)
                metrics_info['SSIM'].append(ssim_slice)
                metrics_info['VOLUME'].append(tgt_file.name)
                metrics_info['SLICE'].append(index)
        #break

    return metrics_info

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--target-path', type=pathlib.Path, required=True,
                        help='Path to the ground truth data')
    parser.add_argument('--predictions-path', type=pathlib.Path, required=True,
                        help='Path to reconstructions')
    parser.add_argument('--report-path', type=pathlib.Path, required=True,
                        help='Path to save metrics')

    args = parser.parse_args()

    recons_key = 'img_gt'

    metrics_info = {'VOLUME':[],'SLICE':[],'MSE':[],'NMSE':[],'PSNR':[],'SSIM':[]}

    metrics_info = evaluate(args,recons_key,metrics_info)
    csv_path     = args.report_path / 'metrics.csv'
    df = pd.DataFrame(metrics_info)
    df.to_csv(csv_path)


