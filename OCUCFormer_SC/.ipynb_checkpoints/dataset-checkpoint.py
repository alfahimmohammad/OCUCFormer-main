import pathlib
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch
from skimage import feature
import os 
from utils import npComplexToTorch

class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, acc_factors,dataset_types,mask_types,train_or_valid,mask_path): # acc_factor can be passed here and saved as self variable
        escapes = ''.join([chr(char) for char in range(1, 32)])
        translator = str.maketrans('', '', escapes)
        self.examples = []
        self.mask_path = mask_path.translate(translator) 
        for dataset_type in dataset_types:
            dataroot = os.path.join(root,dataset_type).translate(translator)
            for mask_type in mask_types:
                newroot = os.path.join(dataroot,mask_type,train_or_valid).translate(translator)
                for acc_factor in acc_factors:
                    files = list(pathlib.Path(os.path.join(newroot,'acc_{}'.format(acc_factor)).translate(translator)).iterdir())
                    for fname in sorted(files):
                        with h5py.File(fname,'r') as hf:
                            fsvol = hf['volfs']
                            num_slices = fsvol.shape[2]
                            self.examples += [(fname, slice, acc_factor, mask_type, dataset_type) for slice in range(num_slices)]



    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        
        fname, slice,acc_factor,mask_type, dataset_type = self.examples[i] 
        escapes = ''.join([chr(char) for char in range(1, 32)])
        translator = str.maketrans('', '', escapes)
    
        with h5py.File(fname, 'r') as data:
            key_img = 'img_volus_{}'.format(acc_factor).translate(translator) 
            key_kspace = 'kspace_volus_{}'.format(acc_factor).translate(translator) 

            input_img  = data[key_img][:,:,slice]
            input_kspace  = data[key_kspace][:,:,slice]
            input_kspace = npComplexToTorch(input_kspace)
    
            target = data['volfs'][:,:,slice].astype(np.float64)# converting to double
            mask = np.load(os.path.join(self.mask_path,dataset_type,mask_type,'mask_{}.npy'.format(acc_factor)).translate(translator))
            return torch.from_numpy(input_img), input_kspace, torch.from_numpy(target),torch.from_numpy(mask)

            
class SliceDataDev(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root,acc_factor,dataset_type,mask_type,mask_path):
        escapes = ''.join([chr(char) for char in range(1, 32)])
        translator = str.maketrans('', '', escapes)
        files = list(pathlib.Path(root.translate(translator)).iterdir())
        self.examples = []
        self.mask_path = mask_path.translate(translator) 
        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                fsvol = hf['volfs']
                num_slices = fsvol.shape[2]
                self.examples += [(fname, slice, acc_factor,mask_type,dataset_type) for slice in range(num_slices)]

          

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        
        fname, slice, acc_factor,mask_type, dataset_type = self.examples[i]
        escapes = ''.join([chr(char) for char in range(1, 32)])
        translator = str.maketrans('', '', escapes)
    
        with h5py.File(fname, 'r') as data:

            key_img = 'img_volus_{}'.format(acc_factor).translate(translator) 
            key_kspace = 'kspace_volus_{}'.format(acc_factor).translate(translator) 
            input_img  = data[key_img][:,:,slice]
            input_kspace  = data[key_kspace][:,:,slice]
            input_kspace = npComplexToTorch(input_kspace)
            target = data['volfs'][:,:,slice]
            mask = np.load(os.path.join(self.mask_path,dataset_type,mask_type,'mask_{}.npy'.format(acc_factor)).translate(translator))
            return torch.from_numpy(input_img), input_kspace, torch.from_numpy(target), torch.from_numpy(mask), str(fname.name),slice

class SliceDisplayDataDev(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root,dataset_type,mask_type,acc_factor,mask_path):
        #print (root,dataset_type,mask_type,acc_factor,mask_path)

        escapes = ''.join([chr(char) for char in range(1, 32)])
        translator = str.maketrans('', '', escapes)
        newroot = os.path.join(root,dataset_type,mask_type,'validation','acc_{}'.format(acc_factor)).translate(translator)
        files = list(pathlib.Path(newroot.translate(translator)).iterdir())
        self.examples = []
        self.acc_factor = acc_factor.translate(translator)
        self.dataset_type = dataset_type.translate(translator)

        self.key_img = 'img_volus_{}'.format(self.acc_factor).translate(translator)
        self.key_kspace = 'kspace_volus_{}'.format(self.acc_factor).translate(translator)

        mask_path = os.path.join(mask_path,dataset_type,mask_type,'mask_{}.npy'.format(acc_factor)).translate(translator)
        self.mask_path = mask_path.translate(translator) 

        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                fsvol = hf['volfs']
                num_slices = fsvol.shape[2]
                self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname, slice = self.examples[i]
    
        with h5py.File(fname, 'r') as data:

            input_img  = data[self.key_img][:,:,slice]
            input_kspace  = data[self.key_kspace][:,:,slice]
            input_kspace = npComplexToTorch(input_kspace)
            target = data['volfs'][:,:,slice].astype(np.float64)
            mask = np.load(self.mask_path)
            return torch.from_numpy(input_img), input_kspace, torch.from_numpy(target),mask
 
