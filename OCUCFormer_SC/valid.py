import pathlib
import sys
from collections import defaultdict
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import SliceDataDev
from models import OCUCFormer
import h5py
from tqdm import tqdm

def save_reconstructions(reconstructions, out_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.
    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
    out_dir.mkdir(exist_ok=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)


def create_data_loaders(args):

    data = SliceDataDev(args.data_path,args.acceleration_factor,args.dataset_type, args.mask_type,args.usmask_path)
    #data = SliceDataDev(args.data_path,args.acceleration_factor,args.dataset_type)
    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
    )

    return data_loader


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
#     model = DnCn(args,n_channels=1,nc=5).to(args.device)
    model = OCUCFormer(args).to(args.device)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])
    return model



def run_unet(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(list)
    with torch.no_grad():
        for (iter,data) in enumerate(tqdm(data_loader)):

            input, input_kspace, target,mask,fnames, slices = data
            input = input.unsqueeze(1).to(args.device)
            input_kspace = input_kspace.to(args.device)
            target = target.unsqueeze(1).to(args.device)
            mask = mask.to(args.device)
            
            input = input.float()

            recons = model(input, input_kspace, mask)  
            
            """
            Kepp only one of the below lines uncommented.
            """
            recons = recons.to('cpu').squeeze(1) #Normal validation 
            for i in range(recons.shape[0]):
#                 recons[i] = recons[i] 
                reconstructions[fnames[i]].append((slices[i].numpy(), recons[i].numpy()))

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }
    return reconstructions


def main(args):
    
    data_loader = create_data_loaders(args)
    model = load_model(args.checkpoint)
    reconstructions = run_unet(args, model, data_loader)
    save_reconstructions(reconstructions, args.out_dir)


def create_arg_parser():

    parser = argparse.ArgumentParser(description="Valid setup for MR recon U-Net")
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True,
                        help='Path to the U-Net model')
    parser.add_argument('--out-dir', type=pathlib.Path, required=True,
                        help='Path to save the reconstructions to')
    parser.add_argument('--batch-size', default=16, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda:0', help='Which device to run on')
    parser.add_argument('--data-path',type=str,help='path to validation dataset')

    parser.add_argument('--acceleration_factor',type=str,help='acceleration factors')
    parser.add_argument('--dataset_type',type=str,help='cardiac,kirby')
    parser.add_argument('--usmask_path',type=str,help='undersampling mask path')
    parser.add_argument('--mask_type',type=str,help='mask type - cartesian, gaussian')
    
    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
