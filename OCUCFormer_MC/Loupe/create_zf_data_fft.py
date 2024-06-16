import h5py
import glob
import os
import numpy as np

h5_dir  = '/media/htic/NewVolume1/murali/MR_reonstruction/kirby_recon_dataset_copy/validation'
save_path = ''

h5_files = glob.glob(os.path.join(h5_dir,'*.h5'))
print (h5_files)

for h5_path in h5_files:
    filename = os.path.basename(h5_path)
    with h5py.File(h5_path,'r') as hf:
        #print(hf.keys())
        target = hf['volfs'].value

    with h5py.File(os.path.join(save_path,filename),'w') as hf:
        zf = np.transpose(zf,[2,0,1])
        hf.create_dataset('reconstruction',data=zf)



