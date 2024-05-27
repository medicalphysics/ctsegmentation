import os
import pandas
import numpy as np
import concurrent
import threading
import queue
import random
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from segmentation_volumes import VOLUMES
import nibabel
import math
from matplotlib import pylab as plt


class TotSegDataset2D(Dataset):
    def __init__(self, data_dir: str, train=True, batch_size=4, volumes=None, rewrite_labels = False , dtype=torch.float32):  
        self._train_shape = (256, 256)     
        self._batch_size = batch_size   
        self._dtype = dtype

        df = pandas.read_csv(os.path.join(data_dir, "meta.csv"), sep=';')
        if train:
            patients = df[df['split'] == 'train']['image_id']
        else:
            patients = df[df['split'] != 'train']['image_id']
        self._volumes = dict()
        if volumes is not None:
            self._volumes = {k+1:v for k, v in enumerate(volumes)}
        self._item_paths = list([os.path.join(data_dir, p) for p in patients if os.path.isdir(os.path.join(data_dir, p))])
        if rewrite_labels:
            self.prepare_labels(True)

        self._item_splits = self._calculate_data_splits(volumes)
        self.max_labels = 117 if volumes is None else len(volumes)
        self._label_tensor_dim=2 
        while self._label_tensor_dim < self.max_labels:
            self._label_tensor_dim*=2 

    def shuffle(self):
        random.shuffle(self._item_splits)              
   
    def _calculate_data_splits(self, volumes=None): 
        shape = [nibabel.load(os.path.join(p, "ct.nii.gz")).shape for p in self._item_paths]        

        splits = list()
        batch = list()
        min_pad_size = min(self._train_shape)//8
        for ind, sh in enumerate(shape):
            x=0                
            while x+min_pad_size < sh[0]:
                y=0
                while y+min_pad_size < sh[1]:
                    z=0
                    while z < sh[2]:                        
                        batch.append((ind, x, y, z))                                                
                        z+=1
                        if len(batch) == self._batch_size:
                            splits.append(batch)
                            batch=list()
                    y += self._train_shape[1]
                x += self._train_shape[0]
            x, y, z = sh
        if len(batch) > 0:
            splits.append(batch)


        if volumes is not None: 
            print("Preparing volumes")           
            ssplits = list()            
            cpat = None
            for batch in tqdm(splits, leave=False):
                for split in batch:
                    pat, xb, yb, zb = split
                    if cpat != pat:
                        cpat = pat                        
                        label = np.asarray(nibabel.load(os.path.join(self._item_paths[cpat], "labels.nii.gz")).dataobj)                    
                    xe = min(label.shape[0], xb+self._train_shape[0])
                    ye = min(label.shape[1], yb+self._train_shape[1])
                    sub_label = label[xb:xe, yb:ye, zb]
                    if any(v in sub_label for v in volumes):
                        ssplits.append((pat, xb, yb, zb))

            splits.clear()
            batch = list()
            for split in ssplits:
                batch.append(split)
                if len(batch) == self._batch_size:
                    splits.append(batch)
                    batch = list()
            if len(batch) != 0:
                splits.append(batch)            

        return splits

    def _load_labels(self, path, overwrite=False):
        label_exists = os.path.exists(os.path.join(path, "labels.nii.gz"))        
    
        if not label_exists or overwrite:            
            d = list([(key, name) for key, name in VOLUMES.items()])
            d.sort(key = lambda x: x[0])
            ct = nibabel.load(os.path.join(path, "ct.nii.gz"))
            arr = np.zeros(ct.shape, dtype=np.uint8)
            for ind, (key, name) in enumerate(tqdm(d, leave=False)):
                sub = np.asarray(nibabel.load(os.path.join(path, "segmentations", "{}.nii.gz".format(name))).dataobj, dtype=None)
                arr[sub.nonzero()]=ind+1
            #arr = np.array([nibabel.load(os.path.join(path, "segmentations", "{}.nii.gz".format(name))).get_fdata().astype(np.uint8) for key, name in d], dtype=np.uint8)            
            im = nibabel.Nifti1Image(arr, ct.affine)
            im.set_data_dtype(np.uint8)
            nibabel.save(im, os.path.join(path, "labels.nii.gz"))           
            return True
        return False

    def prepare_labels(self, overwrite=False):
        """Make labels array"""
        print("Making labels arrays")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:                        
            pbar = tqdm(total=len(self._item_paths))
            futures = list([executor.submit(self._load_labels, p, overwrite) for p in self._item_paths])
            for res in concurrent.futures.as_completed(futures):            
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
     
    @staticmethod
    def _iter_worker(data, queue):
        n_cont = os.cpu_count()//2      
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_cont) as executor:  
        #with concurrent.futures.ProcessPoolExecutor(max_workers=n_cont) as executor:            
            for i in range(len(data)):
                queue.put(executor.submit(data.__getitem__, i))        
        queue.join()

    def __iter__(self):        
        q = queue.Queue(maxsize=os.cpu_count()//2)    
        t = threading.Thread(target=self._iter_worker, args=(self, q))
        t.start()
        for _ in range(len(self)):
            f = q.get().result()
            q.task_done()                    
            yield f        
        t.join()
        
    def get_volumes(self):
        vols = dict()
        for i in range(1, 118):
            vols[i]=0
        for path in tqdm(self._item_paths):
            try:
                _ = os.path.exists(os.path.join(path, "labels.nii.gz"))
            except:
                raise ValueError("Label do not exists")
            else:
                vol = nibabel.load(os.path.join(path, "labels.nii.gz")).get_fdata().astype(np.uint8)
                for key in list(vols.keys()):
                    vols[key]+= (vol == key).sum()
        return vols

    def __getitem__(self, idx):
        batch = self._item_splits[idx]
        image = torch.full((self._batch_size, 1) + self._train_shape,-1024, dtype=self._dtype)        
        label_idx = torch.zeros((self._batch_size, 1) + self._train_shape, dtype=torch.uint8)
        with torch.no_grad():            
            #label_idx = torch.zeros((self._batch_size, 1) + self._train_shape, dtype=torch.int64)
            pat = ""
            for ind, (pat_id, xbeg, ybeg, zbeg) in enumerate(batch):
                if self._item_paths[pat_id] != pat:
                    pat = self._item_paths[pat_id]                    
                    ct = np.asarray(nibabel.load(os.path.join(pat, "ct.nii.gz")).dataobj)                    
                    seg = np.asarray(nibabel.load(os.path.join(pat, "labels.nii.gz")).dataobj)
                xend = min(xbeg + self._train_shape[0], ct.shape[0])
                yend = min(ybeg + self._train_shape[1], ct.shape[1])
                # we are in cpu land so .numpy() is OK
                image.numpy()[ind, 0, 0:xend-xbeg, 0:yend-ybeg] = np.squeeze(ct[xbeg:xend, ybeg:yend, zbeg])
                label_idx.numpy()[ind, 0, 0:xend-xbeg, 0:yend-ybeg] = np.squeeze(seg[xbeg:xend, ybeg:yend, zbeg])
            image.add_(1024.0)
            image.divide_(2048)
            image.clamp_(min=0, max=1)            
            if len(self._volumes) > 0:
                label_corr = torch.zeros_like(label_idx)                
                for ind, v in self._volumes.items():
                    label_corr.masked_fill_(label_idx == v, ind)                    
                label_idx=label_corr
                        

            #label = torch.zeros((self._batch_size, self._label_tensor_dim) + self._train_shape, dtype=self._dtype)
            #label.scatter_(1, label_idx, 1)            
            
        return image, label_idx


    def del_labels(self):
        for pat in tqdm(self._item_paths):
            dp = os.path.join(pat, 'labels.nii.gz')
            if os.path.exists(dp):
                os.remove(dp)

 
    def patient_segments(self):
        data = list()
        for s in self._item_splits:
            for b in s:
                data.append((b[0], b[1], b[2]))
        data = list(set(data))
        data.sort(key=lambda x: x[2])
        data.sort(key=lambda x: x[1])
        data.sort(key=lambda x: x[0])
        return data

if __name__ == '__main__':        
    d = TotSegDataset2D(r"/home/erlend/Totalsegmentator_dataset_v201/", train=False, volumes = [10,11,12,13,14], batch_size=8)
    #d = TotSegDataset2D(r"/home/erlend/Totalsegmentator_dataset_v201/", train=False, batch_size=8)
    #d = TotSegDataset2D(r"D:\totseg\Totalsegmentator_dataset_v201", train=True, batch_size=4)
    #d = TotSegDataset2D(r"C:\Users\ander\totseg", train=False, volumes = [10,11,12,13,14], rewrite_labels=False, batch_size=8)
    
    #d._load_labels(os.path.join(r"D:\totseg\Totalsegmentator_dataset_v201", "s0001"))
    #d.prepare_labels()
    #d.del_labels()

    
    #t = TotSegDataset(r"D:\totseg\Totalsegmentator_dataset_v201", train=True)
    #t.prepare_labels()
    #t.del_labels()    
    
    
    
    if True:
        d.shuffle()
        for imIdx in range(len(d)):
            image, label = d[imIdx]
            
            if len(image.shape) == 5:
                for i in range(d._batch_size):
                    plt.subplot(2, d._batch_size, i+1)
                    plt.imshow(image[i,0,:,:,image.shape[4]//2], cmap='bone', vmin=0,vmax=1)
                for i in range(d._batch_size):
                    plt.subplot(2, d._batch_size, d._batch_size+i+1)
                    plt.imshow(label[i,0,:,:,image.shape[4]//2], vmin=0,vmax=1)
                plt.show()
            else:
                with torch.no_grad():
                    label_index = label.mul(torch.arange(label.shape[1]).reshape((label.shape[1], 1, 1))).sum(dim=1, keepdim=True)        
                for i in range(d._batch_size):
                    print(image[i,0,:,:].max(), image[i,0,:,:].min(), label[i,0,:,:].min(), label[i,0,:,:].max())
                    plt.subplot(2, d._batch_size, i+1)
                    plt.imshow(image[i,0,:,:], cmap='bone', vmin=0,vmax=1)
                for i in range(d._batch_size):
                    plt.subplot(2, d._batch_size, d._batch_size+i+1)
                    #plt.imshow(label_index[i,0,:,:])
                    plt.imshow(label[i,0,:,:])
                plt.tight_layout()
                plt.tight_layout()
                plt.show()
            