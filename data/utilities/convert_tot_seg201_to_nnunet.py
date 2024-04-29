import pandas as pd
import os
import nibabel
from tqdm import tqdm
import numpy as np
import shutil
from matplotlib import pylab as plt

from segmentation_volumes import VOLUMES

TOTSEG_FOLDERPATH = r"C:\Users\ander\totseg\Totalsegmentator_dataset_v201"
NNUNET_FOLDERPATH = r"C:\Users\ander\totseg\nnunet_data"

def read_subjects():
    meta = pd.read_csv(os.path.join(TOTSEG_FOLDERPATH, "meta.csv"), sep=';')
    data = list()
    for _, row in meta.iterrows():
        data.append((row['image_id'], row['split']))
    return data

def test_segmentation_files():
    data = dict()
    all_names = list()

    subjects = read_subjects()
    for sub, _ in subjects:
        subject_path = os.path.join(TOTSEG_FOLDERPATH, sub, "segmentations")
        data[sub] = os.listdir(subject_path)        
        all_names += data[sub]

    all_names = list(set(all_names))

    named_volumes = list([v for v in VOLUMES.values()])
    all_volumes = [a[:-7] for a in all_names]
    for fname in all_volumes:
        for pname in named_volumes:
            if fname not in named_volumes:
                print("Error in named volume {}".format(fname))
            if pname not in all_volumes:
                print("Error in named volume {}".format(pname))

    success = True
    for sub, names in data.items():
        for n in names:
            if n not in all_names:
                print("Missing {} for subject {}".format(n, sub))                
                success = False
        for n in all_names:
            if n not in names:
                print("Missing {} for subject {}".format(n, sub))
                success = False
    if success:
        print("Found all segmentations for all subjects")
    else:
        print("Dateset error, missing segmentations")

def test_segmentation_overlap():
    subjects = read_subjects()
    all_names = list()
    for sub, _ in subjects:
        subject_path = os.path.join(TOTSEG_FOLDERPATH, sub, "segmentations") 
        data = os.listdir(subject_path)       
        all_names += data

    all_names = list(set(all_names))

    for sub, _ in subjects:
        subject_path = os.path.join(TOTSEG_FOLDERPATH, sub, "segmentations")        
        imfile = os.path.join(TOTSEG_FOLDERPATH, sub, "ct.nii.gz")
        ct = nibabel.load(imfile)
        ct_shape = ct.shape
        #seg_array = np.zeros(ct_shape)

        segs = dict()
        for ind, segname in tqdm(enumerate(all_names)):
            segfile = os.path.join(TOTSEG_FOLDERPATH, sub, "segmentations", segname)
            segs[segname] = nibabel.load(segfile).get_fdata().astype(np.uint8)
        
        for ind1 in range(len(all_names)):
            name1 = all_names[ind1]
            arr1 = segs[name1]
            for ind2 in range(ind1+1, len(all_names)):
                name2 = all_names[ind2]                
                arr2 = segs[name2]
                comb = arr1*arr2
                nvox = len(comb[comb > 0])
                if nvox != 0:
                    print("{} overlap {} with {} voxels".format(name1, name2, nvox))
        return





def create_nnunet_dataset():
    subjects = read_subjects()  
      
    for subject, split in subjects:
        subject_path = os.path.join(NNUNET_FOLDERPATH, )









if __name__ == '__main__':
    test_segmentation_files()
    #test_segmentation_overlap()