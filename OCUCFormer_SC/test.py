import pathlib
import sys
from collections import defaultdict
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasetnew import SliceData,taskdataset
from modelsnew import five_layerCNN_MAML
import h5py
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter
import time
from torch.nn import functional as F
import logging
import shutil
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_reconstructions(reconstructions,out_dir,h5_file_name):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.
    Args:
        reconstructions (Model prediction itself of size batchxheight,weight)
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
    #out_dir = pathlib.Path(out_dir_str)
    #out_dir.mkdir(exist_ok=True)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else: 
        shutil.rmtree(out_dir)
        os.makedirs(out_dir)
    file_name = h5_file_name.split("/")[-1]
    with h5py.File(out_dir+file_name, 'w') as f:
        f.create_dataset('reconstruction',data = reconstructions)

        
        
def CreateLoadersForTestTasks(args):
    
    support_loaderdict = {}
    query_loaderdict = {}

    full_tasks = args.test_task_strings.split(",")

    all_dataset_types = []
    all_mask_types = []
    all_acc_types = []
                
    for one_task in full_tasks:
        types = one_task.split('_')
        all_dataset_types.append(types[0] + "_" + types[1] + "_" + types[2])
        all_mask_types.append(types[3])
        all_acc_types.append(types[4])

    dataset_types = np.unique(all_dataset_types)
    mask_types = np.unique(all_mask_types)
    acc_factors = np.unique(all_acc_types)

    test_path = args.test_path
    for dataset_type in dataset_types:
        for mask_type in mask_types:
            for acc_factor in acc_factors:

                support_task_dataset = SliceData(test_path,acc_factor,dataset_type,mask_type,"valid_support")
                query_task_dataset = SliceData(test_path,acc_factor,dataset_type,mask_type,"valid_query")
                
                support_task_loader = DataLoader(dataset = support_task_dataset, batch_size = args.test_support_batch_size,shuffle = True)
                query_task_loader = DataLoader(dataset = query_task_dataset, batch_size = len(query_task_dataset),shuffle = False)
                
                loaderkey = dataset_type+"_"+mask_type+"_"+acc_factor
                
                support_loaderdict[loaderkey] = support_task_loader
                
                query_loaderdict[loaderkey] = query_task_loader
    return support_loaderdict,query_loaderdict


def load_model(checkpoint_file):

    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']

    model = five_layerCNN_MAML(args).to(args.device).double() # to make the weights in double since input type is double 
    model.load_state_dict(checkpoint['model'])

    return model


def test_time_inference(args,model,test_task_loader,test_support_loader,test_query_loader,writer):
    
    start = start_iter = time.perf_counter()
    
    for iter,test_task_mini_batch in enumerate(test_task_loader):
        
        test_meta_loss = 0
        
        for test_task in test_task_mini_batch:
            
            base_weights = list(model.parameters())
            clone_weights = [w.clone() for w in base_weights]
            
            adapt_flag = False
            no_of_gd_steps = 0
            while True:

                for gd_steps,test_spt_data_batch in enumerate(test_support_loader[test_task]):
                    no_of_gd_steps = no_of_gd_steps+1 # gd_steps will reset to 0 once the loader exhausts. Hence we introduce another counter no_of_gd_steps to get the actual number of times the loader is enumerated

                    model.train()

                    test_spt_us_imgs = test_spt_data_batch[0].unsqueeze(1).to(args.device)
                    test_spt_ksp_imgs = test_spt_data_batch[1].to(args.device)
                    test_spt_fs_imgs = test_spt_data_batch[2].unsqueeze(1).to(args.device)
                    test_spt_ksp_mask = test_spt_data_batch[6].to(args.device)
                    test_spt_acc_string = test_spt_data_batch[3][0]
                    test_spt_mask_string = test_spt_data_batch[4][0]
                    test_spt_dataset_string = test_spt_data_batch[5][0]

                    test_spt_fs_pred = model.adaptation(test_spt_us_imgs,clone_weights,test_spt_ksp_imgs,test_spt_ksp_mask)
                    test_spt_loss = F.l1_loss(test_spt_fs_pred,test_spt_fs_imgs)
                    
                    if no_of_gd_steps == 1:
                        best_adapt_loss = test_spt_loss.item()
                        best_adapt_model_weights = [w.clone() for w in clone_weights]
                    elif best_adapt_loss > test_spt_loss.item():
                        best_adapt_loss = test_spt_loss.item()
                        best_adapt_model_weights = [w.clone() for w in clone_weights]
                    
                    clone_grads = torch.autograd.grad(test_spt_loss, clone_weights)
                    clone_weights = [w-args.test_adapt_lr*g for w, g in zip(clone_weights,clone_grads)]
                    
                    if iter % args.report_interval == 0:
                        logging.info(
                                f'Adaptation_step = [{no_of_gd_steps:3d}/{args.no_of_test_adaptation_steps:3d}] '
                                f'Adaptation loss = {test_spt_loss.item():.4g} '
                                f'Time = {time.perf_counter() - start_iter:.4f}s',
                                )
                    start_iter = time.perf_counter()

                    writer.add_scalar(test_task+'_per_adaptation_step_spt_loss',test_spt_loss.item(),no_of_gd_steps)

                    if no_of_gd_steps >= args.no_of_test_adaptation_steps:
                        adapt_flag = True
                        break
                    
                if adapt_flag:
                    print("Adaptation stopped after gd_steps {}".format(no_of_gd_steps))
                    break

            for _,test_qry_data_batch in enumerate(test_query_loader[test_task]):

                model.eval()
                test_qry_us_imgs = test_qry_data_batch[0].unsqueeze(1).to(args.device)
                test_qry_ksp_imgs = test_qry_data_batch[1].to(args.device)
                test_qry_fs_imgs = test_qry_data_batch[2].unsqueeze(1).to(args.device)
                test_qry_ksp_mask = test_qry_data_batch[6].to(args.device)
                test_qry_acc_string = test_qry_data_batch[3][0]
                test_qry_mask_string = test_qry_data_batch[4][0]
                test_qry_dataset_string = test_qry_data_batch[5][0]
                test_qry_file_name = test_qry_data_batch[7][0]
                break
            
            with torch.no_grad(): # this is added to ensure that gradients do not occupy the gpu mem
                test_qry_fs_pred = model.adaptation(test_qry_us_imgs,best_adapt_model_weights,test_qry_ksp_imgs,test_qry_ksp_mask)
                test_qry_loss = F.l1_loss(test_qry_fs_pred,test_qry_fs_imgs)
            
            test_meta_loss = test_meta_loss+test_qry_loss
        
        current_task_path = test_task.split("_")
        task_specific_path = str(args.results_dir) + "/" + current_task_path[0] + "_" + current_task_path[1] + "_" + current_task_path[2] + "/"+current_task_path[3]+"/"+"acc_"+current_task_path[4] + "/"
        
        save_reconstructions(test_qry_fs_pred.squeeze(1).cpu().detach().numpy(),task_specific_path,test_qry_file_name)

        writer.add_scalar(test_task+'_qry_loss',test_meta_loss.item(),1)
        
        logging.info('Query loss for task {}:{}'.format(test_task,test_meta_loss.item()))
    
    return test_qry_fs_pred,time.perf_counter()-start


def main(args):
    
    test_support_loader,test_query_loader = CreateLoadersForTestTasks(args)

    test_task_strings = args.test_task_strings.split(",")

    test_task_dataset_instantiation = taskdataset(test_task_strings)

    test_task_batch_size = args.test_task_batch_size
    test_task_loader = DataLoader(dataset = test_task_dataset_instantiation, batch_size = test_task_batch_size,shuffle = True)

    print ("Task, Support and Query loaders are initialized")
    
    model = load_model(args.checkpoint)
    
    writer = SummaryWriter(log_dir=str(args.tensorboard_summary_dir / 'unseen_test_summary'))
    
    reconstructions,_ = test_time_inference(args,model,test_task_loader,test_support_loader,test_query_loader,writer)

    
    
def create_arg_parser():

    parser = argparse.ArgumentParser(description="Valid setup for MR recon U-Net")
    
    parser.add_argument('--seed',default=42,type=int,help='Seed for random number generators')

    parser.add_argument('--checkpoint', type=pathlib.Path, required=True,
                        help='Path to the trained model')
    parser.add_argument('--results_dir', type=pathlib.Path, required=True,
                        help='Base path to save the reconstructions to')
    parser.add_argument('--tensorboard_summary_dir', type=pathlib.Path, required=True,
                        help='Path to write the summary files')
    
    parser.add_argument('--usmask_path',type=str,help='us mask path') 
    
    parser.add_argument('--test_task_batch_size',default=1,type=int, help='Test task batch size')
    parser.add_argument('--test_support_batch_size', default=47, type=int, help='Support batch size')
#     parser.add_argument('--test_query_batch_size', default=60,type=int, help='Query batch size')
    
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')
    parser.add_argument('--test_path',type=pathlib.Path ,required=True, help = 'path to test dataset')

    parser.add_argument('--no_of_test_adaptation_steps', type=int, default=3,
                        help='Number of adaptation steps during meta-training stage')
    
    parser.add_argument('--test_task_strings',type=str,help='All the test tasks')
    
    parser.add_argument('--test_adapt_lr', type=float, default=0.001, help='Task-specific Learning rate')

    parser.add_argument('--report-interval', type=int, default=1, help='Period of loss reporting')
    return parser

if __name__ == '__main__':
    #args = create_arg_parser().parse_args(sys.argv[1:])
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
