import torch
import numpy as np
import os
import nibabel as nib
from torch.utils import data
from skimage.transform import resize


#############
# Older Dataset class for performing training without any pre-processing. Please see preprocess.py and data_loaders.py for updated version
##########################

# Dataset class that feeds data into a data generator. Takes in a list of folder paths (full path including the folder itself) and folder names of the folders that each contain the multimodal scans
class Dataset(data.Dataset):
    def __init__(self, folder_paths, folder_IDS):
        self.folder_path = folder_paths
        self.folder_ID = folder_IDS

    def __len__(self):
        return len(self.folder_ID)

    def __getitem__(self, index):
        data_folder = self.folder_path[index]
        data_id = self.folder_ID[index]
        img_t1 = resize(nib.load(os.path.join(data_folder, data_id) + "_t1.nii.gz").get_fdata(), (128, 128, 80))
        img_t1ce = resize(nib.load(os.path.join(data_folder, data_id) + "_t1ce.nii.gz").get_fdata(), (128, 128, 80))
        img_t2 = resize(nib.load(os.path.join(data_folder, data_id) + "_t2.nii.gz").get_fdata(), (128, 128, 80))
        img_flair = resize(nib.load(os.path.join(data_folder, data_id) + "_flair.nii.gz").get_fdata(), (128, 128, 80))
        img_segm = resize(nib.load(os.path.join(data_folder, data_id) + "_seg.nii.gz").get_fdata(), (128, 128, 80))

        X = torch.from_numpy(np.asarray([img_t1, img_t1ce, img_t2, img_flair])).float()
        y = torch.from_numpy(img_segm).long()
        return X, y

