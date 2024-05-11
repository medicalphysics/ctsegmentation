import os
import numpy as np
import concurrent
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from segmentation_volumes import VOLUMES
import nibabel

from matplotlib import pylab as plt

class CustomImageDataset(Dataset):
    def __init__(self, data_dir: str):        
        patients = os.listdir(data_dir)
        self._item_paths = list([os.path.join(data_dir, p) for p in patients if os.path.isdir(os.path.join(data_dir, p))])
#        self.prepare_labels()



    def load_labels(self, path, overwrite=False):
        if not os.path.exists(os.path.join(path, "labels.nii.gz")):
            if overwrite:
                ct = nibabel.load(os.path.join(path, "ct.nii.gz"))
                labels = np.zeros(ct.shape, dtype=np.float32)
                for key, name in tqdm(VOLUMES.items(), leave=False):
                    lpart = nibabel.load(os.path.join(path, "segmentations", "{}.nii.gz".format(name))).get_fdata()
                    labels[lpart > 0] = key
                nibabel.save(nibabel.Nifti1Image(labels, ct.affine), os.path.join(path, "labels.nii.gz")) 


    def prepare_labels(self, overwrite=False):
        """Make labels array"""
        print("Making labels arrays")
        

        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()-2) as executor:                        
            pbar = tqdm(total=len(self._item_paths))
            futures = list([executor.submit(self.load_labels, p, overwrite) for p in self._item_paths])
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)
            pbar.close()

    def __len__(self):
        return len(self._item_paths)

    def __getitem__(self, idx):
        image = torch.as_tensor(nibabel.load(os.path.join(self._item_paths[idx], "ct.nii.gz")).get_fdata())
        self.load_labels(self._item_paths[idx])
        label = torch.as_tensor(nibabel.load(os.path.join(self._item_paths[idx], "labels.nii.gz")).get_fdata())        
        return image, label
    
if __name__ == '__main__':
    d = CustomImageDataset(r"/home/erlend/Totalsegmentator_dataset_v201")
    for image, label in d:
        print (image.shape, image.spa)