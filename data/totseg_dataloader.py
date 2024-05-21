import os
import pandas
import numpy as np
import concurrent
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
    def __init__(self, data_dir: str, train=True, batch_size=4):  
        self._train_shape = (256, 256)     
        self._batch_size = batch_size   
        #patients = os.listdir(data_dir)
        df = pandas.read_csv(os.path.join(data_dir, "meta.csv"), sep=';')
        if train:
            patients = df[df['split'] == 'train']['image_id']
        else:
            patients = df[df['split'] != 'train']['image_id']

        self._item_paths = list([os.path.join(data_dir, p) for p in patients if os.path.isdir(os.path.join(data_dir, p))])
        self._item_splits = self._calculate_data_splits()  
        self.max_labels = 117
        self._label_tensor_dim=2
        while self._label_tensor_dim < self.max_labels:
            self._label_tensor_dim*=2 

    def shuffle(self):
        random.shuffle(self._item_splits)              
   
    def _calculate_data_splits(self): 
        shape = [nibabel.load(os.path.join(p, "ct.nii.gz")).shape for p in self._item_paths]        

        splits = list()
        batch = list()
        for ind, sh in enumerate(shape):
            x=0
            while x < sh[0]:
                y=0
                while y < sh[1]:
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
        return splits

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
            d = list([(key, name) for key, name in VOLUMES.items()])
            d.sort(key = lambda x: x[0])
            first = nibabel.load(os.path.join(path, "segmentations", "{}.nii.gz".format(d[0][1])))
            arr = np.zeros(first.shape, dtype=np.uint8)
            for ind, (key, name) in enumerate(tqdm(d, leave=False)):
                sub = nibabel.load(os.path.join(path, "segmentations", "{}.nii.gz".format(name))).get_fdata().astype(np.uint8)
                arr[sub.nonzero()]=ind+1
            #arr = np.array([nibabel.load(os.path.join(path, "segmentations", "{}.nii.gz".format(name))).get_fdata().astype(np.uint8) for key, name in d], dtype=np.uint8)            
            im = nibabel.Nifti1Image(arr, first.affine)
            im.set_data_dtype(np.uint8)
            nibabel.save(im, os.path.join(path, "labels.nii.gz"))           
            return True
        return False

    def prepare_labels(self, overwrite=False):
        """Make labels array"""
        print("Making labels arrays")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()*2) as executor:                        
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
     


    def __iter__(self):  
        n_cont = 24      
        q = queue.Queue(maxsize=n_cont*2)
        def start_jobs(queue, dataset):
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_cont) as executor2:  
                for i in range(len(dataset)):
                    job = executor2.submit(dataset.__getitem__, i)
                    q.put(job)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as exe:
            exe.submit(start_jobs, q, self)
            for i in range(len(self)):
                f = q.get()
                yield f.result()
                q.task_done()
        q.join()

    
        
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
        image = torch.zeros((self._batch_size, 1) + self._train_shape, dtype=torch.float32)
        label = torch.zeros((self._batch_size, self._label_tensor_dim) + self._train_shape, dtype=torch.bool)
        with torch.no_grad():
            label_idx = torch.zeros((self._batch_size, 1) + self._train_shape, dtype=torch.int64)
            pat = ""
            for ind, (pat_id, xbeg, ybeg, zbeg) in enumerate(batch):
                if self._item_paths[pat_id] != pat:
                    pat = self._item_paths[pat_id]
                    ct = nibabel.load(os.path.join(pat, "ct.nii.gz"))        
                    seg = nibabel.load(os.path.join(pat, "labels.nii.gz"))                
                xend = min(xbeg + self._train_shape[0], ct.shape[0])
                yend = min(ybeg + self._train_shape[1], ct.shape[1])
                # we are in cpu land so .numpy() is OK
                image.numpy()[ind, 0, :xend-xbeg, :yend-ybeg] = np.squeeze(ct.slicer[xbeg:xend, ybeg:yend, zbeg:zbeg+1].get_fdata().astype(np.float32))
                label_idx.numpy()[ind, 0, :xend-xbeg, :yend-ybeg] = np.squeeze(seg.slicer[xbeg:xend, ybeg:yend, zbeg:zbeg+1].get_fdata().astype(np.int64))
            label.scatter_(1, label_idx, 1)
            image.subtract_(1024.0)
            image.divide_(2048)
            image.clamp_(min=0, max=1)
        return image, label


    def del_labels(self):
        for pat in tqdm(self._item_paths):
            dp = os.path.join(pat, 'labels.nii.gz')
            if os.path.exists(dp):
                os.remove(dp)

    

class TotSegDataset3D(Dataset):
    def __init__(self, data_dir: str, train=True, batch_size=4):  
        self._train_shape = (128, 128, 64)     
        self._batch_size = batch_size   
        #patients = os.listdir(data_dir)
        df = pandas.read_csv(os.path.join(data_dir, "meta.csv"), sep=';')
        if train:
            patients = df[df['split'] == 'train']['image_id']
        else:
            patients = df[df['split'] != 'train']['image_id']

        self._item_paths = list([os.path.join(data_dir, p) for p in patients if os.path.isdir(os.path.join(data_dir, p))])
        self._item_splits = self._calculate_data_splits()  
        self.max_labels = 117         
   
    def shuffle(self):
        random.shuffle(self._item_splits)

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
            d = list([(key, name) for key, name in VOLUMES.items()])
            d.sort(key = lambda x: x[0])
            first = nibabel.load(os.path.join(path, "segmentations", "{}.nii.gz".format(d[0][1])))
            arr = np.zeros(first.shape, dtype=np.uint8)
            for ind, (key, name) in enumerate(tqdm(d, leave=False)):
                sub = nibabel.load(os.path.join(path, "segmentations", "{}.nii.gz".format(name))).get_fdata().astype(np.uint8)
                arr[sub.nonzero()]=ind+1
            #arr = np.array([nibabel.load(os.path.join(path, "segmentations", "{}.nii.gz".format(name))).get_fdata().astype(np.uint8) for key, name in d], dtype=np.uint8)            
            im = nibabel.Nifti1Image(arr, first.affine)
            im.set_data_dtype(np.uint8)
            nibabel.save(im, os.path.join(path, "labels.nii.gz"))           
            return True
        return False

    def prepare_labels(self, overwrite=False):
        """Make labels array"""
        print("Making labels arrays")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()*2) as executor:                        
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
        return int(math.ceil(len(self._item_splits)/self._batch_size))
     
    def __iter__(self):       
        with Pool(processes=4) as pool:
            for r in pool.imap_unordered(self.__getitem__, range(len(self))):
                yield r
    
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
        if self._batch_size == 1:
            return self._get_batch_part(idx)
        else:
            split_beg = idx * self._batch_size
            if split_beg >= len(self._item_splits):
                raise IndexError        
            split_end = min((idx + 1) * self._batch_size, len(self._item_splits))

            ims = list([self._get_batch_part(i) for i in range(split_beg, split_end)])
            image = torch.cat([k[0] for k in ims])
            labels = torch.cat([k[1] for k in ims])
            return image, labels
 
    
    def _get_batch_part(self, idx):
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
    
    def del_labels(self):
        for pat in tqdm(self._item_paths):
            dp = os.path.join(pat, 'labels.nii.gz')
            if os.path.exists(dp):
                os.remove(dp)



if __name__ == '__main__':        
    d = TotSegDataset2D(r"/home/erlend/Totalsegmentator_dataset_v201/", train=False, batch_size=6)
    #d = TotSegDataset2D(r"D:\totseg\Totalsegmentator_dataset_v201", train=True, batch_size=8)
    #d._load_labels(os.path.join(r"D:\totseg\Totalsegmentator_dataset_v201", "s0001"))
    #d.prepare_labels()
    #d.del_labels()

    
    #t = TotSegDataset(r"D:\totseg\Totalsegmentator_dataset_v201", train=True)
    #t.prepare_labels()
    #t.del_labels()

    
    if True:
        d.shuffle()
        for image, label in d:        
            print(image.shape, label.shape)
            print(image.max(), image.min())
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
                    plt.subplot(2, d._batch_size, i+1)
                    plt.imshow(image[i,0,:,:], cmap='bone', vmin=0,vmax=1)
                for i in range(d._batch_size):
                    plt.subplot(2, d._batch_size, d._batch_size+i+1)
                    plt.imshow(label_index[i,0,:,:])
                plt.tight_layout()
                plt.show()
            