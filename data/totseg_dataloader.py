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
    def __init__(self, data_dir: str, n_labels=None):  
        self._train_shape = (128, 128, 128)        
        patients = os.listdir(data_dir)
        self._item_paths = list([os.path.join(data_dir, p) for p in patients if os.path.isdir(os.path.join(data_dir, p))])
        self._item_splits = self._calculate_data_splits()        
        if n_labels is None:
            self.prepare_labels()
            self._n_labels = self._find_max_label()
        else: 
            self._n_labels = n_labels
        
   
    def _calculate_data_splits(self):        
        shape = [nibabel.load(os.path.join(p, "ct.nii.gz")).shape for p in self._item_paths]        
        splits =  [(s[0]//self._train_shape[0]+1, s[1]//self._train_shape[1]+1,s[2]//self._train_shape[2]+1) for s in shape]
        data = list()
        for ind, spl in enumerate(splits):            
            x, y, z = spl
            for i in range(x):
                    for j in range(y):
                        for k in range(z):                            
                            data.append((ind, (i, j, k)))
        return data


    def _load_labels(self, path, overwrite=False):
        label_exists = os.path.exists(os.path.join(path, "labels.nii.gz"))
        if label_exists and overwrite:
            try:
                _ = nibabel.load(os.path.join(path, "labels.nii.gz")).get_fdata()
            except:
                label_exists = False
            else:
                label_exists = True
        
        if not label_exists:
            ct = nibabel.load(os.path.join(path, "ct.nii.gz"))
            labels = np.zeros(ct.shape, dtype=np.float32)
            for key, name in tqdm(VOLUMES.items(), leave=False):
                lpart = nibabel.load(os.path.join(path, "segmentations", "{}.nii.gz".format(name))).get_fdata()
                labels[lpart.nonzero()] = key
            nibabel.save(nibabel.Nifti1Image(labels, ct.affine), os.path.join(path, "labels.nii.gz")) 

    def prepare_labels(self, overwrite=False):
        """Make labels array"""
        print("Making labels arrays")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()-2) as executor:                        
            pbar = tqdm(total=len(self._item_paths))
            futures = list([executor.submit(self._load_labels, p, overwrite) for p in self._item_paths])
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)
            pbar.close()


    def _find_max_label(self):
        paths = list([os.path.join(p, 'labels.nii.gz') for p in self._item_paths])
        def get_max(path):
            return nibabel.load(path).get_fdata().max()
        gmax = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()-2) as executor:
            pbar = tqdm(total=len(paths))
            futures = list([executor.submit(get_max, p,) for p in paths])
            for max in concurrent.futures.as_completed(futures):
                gmax = max(max, gmax)
                pbar.update(1)
            pbar.close()
        return gmax

    def __len__(self):
        return len(self._item_splits)
     
    def __iter__(self):
        for i in range(len(self._item_splits)):
            yield self.__getitem__(i)

    def __getitem__(self, idx):
        pat, split = self._item_splits[idx]

        image = nibabel.load(os.path.join(self._item_paths[pat], "ct.nii.gz"))
        self._load_labels(self._item_paths[pat], True)
        label = nibabel.load(os.path.join(self._item_paths[pat], "labels.nii.gz"))                
        
        beg = tuple((s*i for s, i in zip(self._train_shape, split)))
        end = tuple((s+i for s, i in zip(beg, self._train_shape)))
        sh = image.shape
        if end[0] > sh[0] or end[1] > sh[1] or end[2] > sh[2]:
            end_p = tuple((min(s, e) for e, s in zip(end, sh)))
            pads = tuple(((0, e-s) for e,s in zip(end, end_p)))
            image_part = np.pad(image.slicer[beg[0]:end_p[0], beg[1]:end_p[1], beg[2]:end_p[2]].get_fdata(), pads, constant_values=-1024)            
            label_part = np.pad(label.slicer[beg[0]:end_p[0], beg[1]:end_p[1], beg[2]:end_p[2]].get_fdata(), pads, constant_values=0)                        
        else:
            image_part = image.slicer[beg[0]:end[0], beg[1]:end[1], beg[2]:end[2]].get_fdata()
            label_part = label.slicer[beg[0]:end[0], beg[1]:end[1], beg[2]:end[2]].get_fdata()
        
        return torch.as_tensor(image_part), torch.as_tensor(label_part)
        
        
        


if __name__ == '__main__':
    d = CustomImageDataset(r"C:\Users\ander\totseg", n_labels=118)
    d.prepare_labels(True)
    print(len(d))
    for image, label in d:         
        plt.subplot(1, 2, 1)
        plt.imshow(image[:,:,image.shape[2]//2], cmap='bone')
        plt.subplot(1, 2, 2)
        plt.imshow(label[:,:,image.shape[2]//2])
        plt.show()
    