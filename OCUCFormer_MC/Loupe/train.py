import sys
import logging
import pathlib
import random
import shutil
import time
import functools
import numpy as np
import argparse

import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataset import SliceData,KneeData
from models import UnetModel
import torchvision
from torch import nn
from torch.autograd import Variable
from torch import optim
from tqdm import tqdm
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_datasets(args):

    train_data = SliceData(args.train_path,args.acceleration_factor,args.dataset_type)
    dev_data = SliceData(args.validation_path,args.acceleration_factor,args.dataset_type)

    return dev_data, train_data

def create_datasets_knee(args):

    train_data = KneeData(args.train_path,args.acceleration_factor,args.dataset_type)
    dev_data = KneeData(args.validation_path,args.acceleration_factor,args.dataset_type)

    return dev_data, train_data



def create_data_loaders(args):

    if args.dataset_type == 'knee':
        dev_data, train_data = create_datasets_knee(args)
    else:
        dev_data, train_data = create_datasets(args)   

    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        #num_workers=64,
        #pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        #num_workers=64,
        #pin_memory=True,
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=16,
        #num_workers=64,
        #pin_memory=True,
    )
    return train_loader, dev_loader, display_loader


def train_epoch(args, epoch, model,data_loader, optimizer, writer,random_mask,prob_mask):

    model.train()
    
    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    #print ("Entering Train epoch")

    for iter, data in enumerate(tqdm(data_loader)):

        #print (data)

        #print ("Received data from loader")
        _,_,fs_img = data # Return kspace also we can ignore that for train and test 
        fs_img = fs_img.unsqueeze(1).float()
        fs_img = fs_img.to(args.device)
        fs_kspace = torch.rfft(fs_img,2,onesided=False)

        slope = 5
        sparsity = 0.5	
        #prob_mask_mean = torch.mean(prob_mask)	

        #print (prob_mask)
        #import pdb
        #pdb.set_trace()

        #print (prob_mask)

        #print (prob_mask_mean)

        #factor1 = sparsity / prob_mask_mean 
        #factor2 = (1 - sparsity) / (1-prob_mask_mean)
        #prob_mask = F.sigmoid(prob_mask)

        #if factor1 < 1:
        #    rescale_probmask = prob_mask * factor1
        #else:
        #    rescale_probmask = 1 - (1 - prob_mask) * factor2 

        #thresh_mask = rescale_probmask > random_mask
        thresh_mask = torch.sigmoid(slope * (prob_mask - random_mask))
        #thresh_mask = thresh_mask.float().unsqueeze(-1).unsqueeze(0)
        thresh_mask = thresh_mask.unsqueeze(-1).float()
        xbar = torch.mean(thresh_mask)
        r = sparsity / xbar
        beta = (1 - sparsity) / (1-xbar)
        le = torch.le(r,1).float()
    
        thresh_mask = le * thresh_mask * r + (1-le) * (1 - (1 - thresh_mask) * beta)

        us_kspace = fs_kspace * thresh_mask
        us_img = torch.ifft(us_kspace,2,True)[:,:,:,:,0]


        output = model(us_img)
        #print ("Input passed to model")
        loss = F.l1_loss(output,fs_img)
        #print ("Loss calculated")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        writer.add_scalar('TrainLoss',loss.item(),global_step + iter )


        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
        #break

    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_loader, writer,random_mask,prob_mask):

    model.eval()
    losses = []
    start = time.perf_counter()
    
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
    
            _,_,fs_img = data # Return kspace also we can ignore that for train and test
            fs_img = fs_img.unsqueeze(1).float().to(args.device)
            fs_kspace = torch.rfft(fs_img,2,onesided=False)
            slope = 5 
            #thresh_mask = prob_mask > random_mask
            thresh_mask = torch.sigmoid(slope * (prob_mask - random_mask))
            #thresh_mask = thresh_mask.float().unsqueeze(-1).unsqueeze(0)
            thresh_mask = thresh_mask.unsqueeze(-1).float()
            us_kspace = thresh_mask * fs_kspace
            us_img = torch.ifft(us_kspace,2,True)[:,:,:,:,0]

            output = model(us_img)
            #loss = F.mse_loss(output,target, size_average=False)
            loss = F.mse_loss(output,fs_img)
            
            losses.append(loss.item())
            #break
            
        writer.add_scalar('Dev_Loss',np.mean(losses),epoch)
       
    return np.mean(losses), time.perf_counter() - start


def visualize(args, epoch, model, data_loader, writer,random_mask,prob_mask):
    
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    def save_mask(mask, tag):
        #print (mask.shape)
        mask = mask[:,:,:,0]
        grid = torchvision.utils.make_grid(mask, nrow=1)
        writer.add_image(tag, grid, epoch)


    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            _,_,fs_img = data # Return kspace also we can ignore that for train and test
            fs_img = fs_img.unsqueeze(1).float().to(args.device)
            fs_kspace = torch.rfft(fs_img,2,onesided=False)
            slope = 5
            #thresh_mask = prob_mask > random_mask 
            thresh_mask = torch.sigmoid(slope * (prob_mask - random_mask))
            thresh_mask = thresh_mask.float().unsqueeze(-1).unsqueeze(0)
            #thresh_mask = thresh_mask.unsqueeze(-1).float()
            us_kspace = fs_kspace * thresh_mask
            us_img = torch.ifft(us_kspace,2,True)[:,:,:,:,0]

            output = model(us_img)
            print("input: ", torch.min(us_img), torch.max(us_img))
            print("target: ", torch.min(fs_img), torch.max(fs_img))
            print("predicted: ", torch.min(output), torch.max(output))
#            print (us_img.shape,fs_img.shape,output.shape,thresh_mask.shape)
            save_image(us_img, 'Input')
            save_image(fs_img, 'Target')
            save_image(output, 'Reconstruction')
            save_mask(thresh_mask, 'Learned mask')
            save_image(torch.abs(fs_img.float() - output.float()), 'Error')
            break

def save_model(args, exp_dir, epoch, model, optimizer,best_dev_loss,is_new_best,prob_mask):

    out = torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'mask': prob_mask,
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir':exp_dir
        },
        f=exp_dir / 'model.pt'
    )

    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def build_model(args):
    
    model = UnetModel(
        in_chans=1,
        out_chans=1,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob
    ).to(args.device)
    
    return model

def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)

    if args.data_parallel:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint, model, optimizer 


def build_optim(args, params):
    optimizer = torch.optim.Adam(params,args.lr, weight_decay=args.weight_decay)
    return optimizer


def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    #writer = SummaryWriter(logdir=str(args.exp_dir / 'summary'))
    writer = SummaryWriter(log_dir=str(args.exp_dir / 'summary'))

    random_mask = torch.rand([240,240],device=args.device)
    prob_mask   = torch.rand([240,240],requires_grad=True,device=args.device)
 
    print (random_mask,prob_mask)

    if args.resume:
        print('resuming model, batch_size', args.batch_size)
        #checkpoint, model, optimizer, disc, optimizerD = load_model(args, args.checkpoint)
        checkpoint, model, optimizer, disc, optimizerD = load_model(args.checkpoint)
        args = checkpoint['args']
        args.batch_size = 28
        best_dev_mse= checkpoint['best_dev_mse']
        best_dev_ssim = checkpoint['best_dev_mse']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        model = build_model(args)
        #print ("Model Built")
        if args.data_parallel:
            model = torch.nn.DataParallel(model)    
        #print (len(list(model.parameters())),prob_mask)
        optim_param = list(model.parameters()) + [prob_mask]
        print (len(optim_param))
        #optimizer = build_optim(args, model.parameters())
        optimizer = build_optim(args, optim_param)
        #print ("Optmizer initialized")
        best_dev_loss = 1e9
        start_epoch = 0

    logging.info(args)
    logging.info(model)

    train_loader, dev_loader, display_loader = create_data_loaders(args)
    #print ("Dataloader initialized")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    
    for epoch in range(start_epoch, args.num_epochs):

        scheduler.step(epoch)
        train_loss,train_time = train_epoch(args, epoch, model,train_loader,optimizer,writer,random_mask,prob_mask)
        dev_loss,dev_time = evaluate(args, epoch, model, dev_loader, writer,random_mask,prob_mask)
        visualize(args, epoch, model, display_loader, writer,random_mask,prob_mask)

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss,dev_loss)
        save_model(args, args.exp_dir, epoch, model, optimizer,best_dev_loss,is_new_best,prob_mask)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g}'
            f'DevLoss= {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )
    writer.close()


def create_arg_parser():

    parser = argparse.ArgumentParser(description='Train setup for MR recon U-Net')
    parser.add_argument('--seed',default=42,type=int,help='Seed for random number generators')
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')
    parser.add_argument('--batch-size', default=2, type=int,  help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--data-parallel', action='store_true', 
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--train-path',type=str,help='Path to train h5 files')
    parser.add_argument('--validation-path',type=str,help='Path to test h5 files')

    parser.add_argument('--acceleration_factor',type=str,help='acceleration factors')
    parser.add_argument('--dataset_type',type=str,help='cardiac,kirby')
    #parser.add_argument('--usmask_path',type=str,help='us mask path')
    
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    #random.seed(args.seed)
    #np.random.seed(args.seed)
    #torch.manual_seed(args.seed)
    #print (args)
    main(args)
