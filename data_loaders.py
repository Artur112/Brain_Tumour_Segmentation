import torch
import numpy as np
import os
import nibabel as nib
from torch.utils import data
from skimage.transform import resize
from sklearn import preprocessing
import sklearn
from matplotlib import pyplot as plt

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

        # Preprocess
        X = []
        i = 1
        for modality in [img_t1, img_t1ce, img_t2, img_flair]:
            brain_region = modality > 0
            mean = np.mean(modality[brain_region])
            stdev = np.std(modality[brain_region])
            new_img = np.zeros((128,128,80))
            new_img[brain_region] = (modality[brain_region] - mean)/stdev
            new_img[new_img > 5] = 5
            new_img[new_img < -5] = -5
            maximum = np.max(new_img)
            minimum = np.min(new_img[brain_region])
            range = maximum - minimum
            new_img = ((((new_img - minimum)/range - 0.5) * 2) + 1)/2
            X.append(new_img)
            
        X = torch.from_numpy(np.asarray(X)).float()
        y = torch.from_numpy(img_segm).long()
        return X, y

