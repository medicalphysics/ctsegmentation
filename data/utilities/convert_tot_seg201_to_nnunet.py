import os
import shutil
import json
from concurrent.futures import ThreadPoolExecutor
import nibabel
from tqdm import tqdm
import numpy as np
import pandas as pd

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

def read_segmentation(path):
    return nibabel.load(path).get_fdata().astype(np.uint8)


def combine_segmentations(ct_path, subject_path, out_path):       
    ct = nibabel.load(ct_path)
    seg_map = np.zeros(ct.shape, dtype=np.uint8)            
    jobs = list()
    with ThreadPoolExecutor(max_workers=64) as executor:
        for key, name in VOLUMES.items():
            path = os.path.join(subject_path, "segmentations", f"{name}.nii.gz")
            jobs.append((key, executor.submit(read_segmentation, path)))
    
    for key, fut in tqdm(jobs, leave=False):
        seg_map[fut.result() > 0] = key 
    nibabel.save(nibabel.Nifti1Image(seg_map, ct.affine), out_path)


def create_json(subjects):        
    subjects_train = list([s for s, split in subjects if split == 'train'])
    subjects_val = list([s for s, split in subjects if split == 'val'])
    
    json_dict = {}
    json_dict['name'] = "CTQA_TotalSegmentator"
    json_dict['channel_names'] = {"0": "CT"}
    
    json_dict['labels'] = {val: idx for idx, val in (VOLUMES | {0:'background'}).items()}
    json_dict['numTraining'] = len(subjects_train) + len(subjects_val)
    json_dict['file_ending'] = '.nii.gz'
    json_dict['overwrite_image_reader_writer'] = 'NibabelIOWithReorient'

    with open(os.path.join(NNUNET_FOLDERPATH, "dataset.json"), "w") as fil:
        json.dump(json_dict, fil, sort_keys=False, indent=4)
    
    splits = []
    splits.append({
        "train": subjects_train,
        "val": subjects_val
    })
    with open(os.path.join(NNUNET_FOLDERPATH, "splits_final.json"), "w") as fil:
        json.dump(splits, fil, sort_keys=False, indent=4)


def create_nnunet_dataset():
    os.makedirs(os.path.join(NNUNET_FOLDERPATH, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(NNUNET_FOLDERPATH, "labelsTr"), exist_ok=True)
    os.makedirs(os.path.join(NNUNET_FOLDERPATH, "imagesTs"), exist_ok=True)
    os.makedirs(os.path.join(NNUNET_FOLDERPATH, "labelsTs"), exist_ok=True)
    subjects = read_subjects()            
    create_json(subjects)
    for subject, split in tqdm(subjects):
        subject_path = os.path.join(TOTSEG_FOLDERPATH, subject)
        ct_path = os.path.join(subject_path, "ct.nii.gz")
        
        ## copy ct image
        if split == "train" or split == "val":            
            shutil.copy(ct_path, os.path.join(NNUNET_FOLDERPATH, "imagesTr", f"{subject}_000.nii.gz"))
            combine_segmentations(ct_path, subject_path, os.path.join(NNUNET_FOLDERPATH, "labelsTr", f"{subject}.nii.gz"))            
        else:
            shutil.copy(ct_path, os.path.join(NNUNET_FOLDERPATH, "imagesTs", f"{subject}_000.nii.gz"))
            combine_segmentations(ct_path, subject_path, os.path.join(NNUNET_FOLDERPATH, "labelsTs", f"{subject}.nii.gz"))            

if __name__ == '__main__':
    #test_segmentation_files()
    #test_segmentation_overlap()
    create_nnunet_dataset()
