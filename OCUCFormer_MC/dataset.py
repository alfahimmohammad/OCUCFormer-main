import pathlib
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
from data import transforms as T
import torch
import os 


class KneeData(Dataset):

    def __init__(self, root):
        
        escapes = ''.join([chr(char) for char in range(1,32)])
        translator = str.maketrans('','',escapes)
        files = list(pathlib.Path(root.translate(translator)).iterdir())
        self.examples = []         
        for fname in sorted(files):
            self.examples.append(fname)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        
        fname = self.examples[i] 

        with h5py.File(fname, 'r') as data:

            img_gt    = torch.from_numpy(data['img_gt'][:])#.value
            img_gt    = T.complex_center_crop(img_gt,(320,320))
            img_und   = torch.from_numpy(data['img_und'][:])#.value
            img_und_kspace = torch.from_numpy(data['img_und_kspace'][:])#.value
            rawdata_und = torch.from_numpy(data['rawdata_und'][:])#.value
            masks = torch.from_numpy(data['masks'][:])#.value
            sensitivity = torch.from_numpy(data['sensitivity'][:])#.value
            """
            img_gt:  torch.Size([640, 368, 2]) img_und:  torch.Size([640, 368, 2]) img_und_kspace:  torch.Size([640, 368, 2]) rawdata_und:  torch.Size([15, 640, 368, 2]) masks:  torch.Size([15, 640, 368, 2]) sensitivity:  torch.Size([15, 640, 368, 2])
            """
            return img_gt,img_und,rawdata_und,masks,sensitivity


class KneeDataDev(Dataset):

    def __init__(self, root):

        escapes = ''.join([chr(char) for char in range(1, 32)])
        translator = str.maketrans('', '', escapes)
        files = list(pathlib.Path(root.translate(translator)).iterdir())
        self.examples = []

        for fname in sorted(files):
            self.examples.append(fname) 

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        
        fname = self.examples[i]
    
        with h5py.File(fname, 'r') as data:

            img_gt    = torch.from_numpy(data['img_gt'][:])#.value
            img_gt    = T.complex_center_crop(img_gt,(320,320))
            img_und   = torch.from_numpy(data['img_und'][:])#.value
            img_und_kspace = torch.from_numpy(data['img_und_kspace'][:])#.value
            rawdata_und = torch.from_numpy(data['rawdata_und'][:])#.value
            masks = torch.from_numpy(data['masks'][:])#.value
            sensitivity = torch.from_numpy(data['sensitivity'][:])#.value
 
       
        return  img_gt,img_und,rawdata_und,masks,sensitivity,str(fname.name)




