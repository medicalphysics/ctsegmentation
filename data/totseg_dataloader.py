import os
import numpy as np
import concurrent
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from segmentation_volumes import VOLUMES
import nibabel
import math
from matplotlib import pylab as plt

class TotSegDataset(Dataset):
    def __init__(self, data_dir: str, max_labels=None, batch_size=4):  
        self._train_shape = (128, 128, 64)     
        self._batch_size = batch_size   
        patients = os.listdir(data_dir)
        self._item_paths = list([os.path.join(data_dir, p) for p in patients if os.path.isdir(os.path.join(data_dir, p))])
        self._item_splits = self._calculate_data_splits()        
        if max_labels is None:
            self.prepare_labels()
            self.max_labels = self._find_max_label()
        else: 
            self.max_labels = max_labels
        
   
    def _calculate_data_splits(self):        
        shape = [nibabel.load(os.path.join(p, "ct.nii.gz")).shape for p in self._item_paths]        
        splits =  [(s[0]//self._train_shape[0], s[1]//self._train_shape[1],s[2]//self._train_shape[2]) for s in shape]
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
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()-1) as executor:                        
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
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()-1) as executor:
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
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def number_of_batches(self):
        return int(math.ceil(self.__len__()/self._batch_size))

    def iter_batch(self):
        idx = 0
        while idx < self.__len__()//self._batch_size:
            ims = [self.__getitem__(idx+i) for i in range(self._batch_size)]                
            image = torch.cat([k[0] for k in ims])
            labels = torch.cat([k[1] for k in ims])
            yield image, labels
            idx += self._batch_size        
        #handle rest
        ims = list([self.__getitem__(i) for i in range(idx, self.__len__())])
        if len(ims) > 0:
            #while len(ims) < 4:
            #    ims.append((torch.zeros_like(ims[0][0]), torch.zeros_like(ims[1][0])))
            image = torch.cat([k[0] for k in ims])
            labels = torch.cat([k[1] for k in ims])
            yield image, labels

    
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
            image_part = np.pad(image.slicer[beg[0]:end_p[0], beg[1]:end_p[1], beg[2]:end_p[2]].get_fdata(), pads, constant_values=-1024).reshape((1, 1) + self._train_shape)
            label_part = np.pad(label.slicer[beg[0]:end_p[0], beg[1]:end_p[1], beg[2]:end_p[2]].get_fdata(), pads, constant_values=0).reshape((1, 1) + self._train_shape)
        else:
            image_part = image.slicer[beg[0]:end[0], beg[1]:end[1], beg[2]:end[2]].get_fdata().reshape((1, 1) + self._train_shape)
            label_part = label.slicer[beg[0]:end[0], beg[1]:end[1], beg[2]:end[2]].get_fdata().reshape((1, 1) + self._train_shape)  
            
        return torch.as_tensor((image_part+1024)/(1024+1024), dtype=torch.float32).clamp_(0,1), torch.as_tensor(label_part/self.max_labels, dtype=torch.float32)
        


if __name__ == '__main__':    
    max_label = max([k for k in VOLUMES.keys()])
    d = TotSegDataset(r"D:\totseg\Totalsegmentator_dataset_v201", max_labels=max_label)
    print("Number of patients: {}, number of  batches {}".format(len(d._item_paths), len(d._item_splits)))
    #d.prepare_labels(True)

    for image, label in d.iter_batch():        
        for i in range(d._batch_size):
            plt.subplot(2, d._batch_size, i+1)
            plt.imshow(image[i,0,:,:,image.shape[2]//2], cmap='bone', vmin=0,vmax=1)
        for i in range(d._batch_size):
            plt.subplot(2, d._batch_size, d._batch_size+i+1)
            plt.imshow(label[i,0,:,:,image.shape[2]//2], vmin=0,vmax=1)
        plt.show()
    