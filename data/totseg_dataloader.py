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
        self._train_shape = (128, 128, 128)        
        patients = os.listdir(data_dir)
        self._item_paths = list([os.path.join(data_dir, p) for p in patients if os.path.isdir(os.path.join(data_dir, p))])
        self._item_splits = self._calculate_data_splits()
        
#        self.prepare_labels()
   
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


    def load_labels(self, path, overwrite=False):
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
        return len(self._item_splits)
     

    def __getitem__(self, idx):
        pat, split = self._item_splits[idx]

        image = nibabel.load(os.path.join(self._item_paths[pat], "ct.nii.gz"))
        self.load_labels(self._item_paths[pat], True)
        label = nibabel.load(os.path.join(self._item_paths[pat], "labels.nii.gz"))        
        print(self._item_paths[pat], image.header.get_zooms(), image.shape)


        
        return torch.as_tensor(image.get_fdata()), torch.as_tensor(label.get_fdata())


if __name__ == '__main__':
    d = CustomImageDataset(r"/home/erlend/Totalsegmentator_dataset_v201")
    for i in range(len(d)):
        print(d._item_splits[i])

    """for image, label in d:
        plt.subplot(1, 2, 1)
        plt.imshow(image[:,:,image.shape[2]//2], cmap='bone')
        plt.subplot(1, 2, 2)
        plt.imshow(label[:,:,image.shape[2]//2])
        plt.show()
       """ 