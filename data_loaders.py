import torch
import numpy as np
import os
from torch.utils import data


##########################################################################
# Dataset class that feeds data into a data generator. Takes in a list of folder paths in which a patients scans are stored, as well
# as a list of the names of the folders / patient ids.
# Returns two torch arrays, one that contains the multi-modal scans and the other that contains the masks.
##########################################################################


class Dataset(data.Dataset):
    def __init__(self, folder_path, folder_id):
        self.folder_paths = folder_path
        self.folder_ids = folder_id

    def __len__(self):
        return len(self.folder_ids)

    def __getitem__(self, index):
        data_folder = self.folder_paths[index]
        data_id = self.folder_ids[index]
        X = np.load(r"{}\\{}_scans.npy".format(os.path.join(data_folder, data_id),data_id))
        y = np.load(r"{}\\{}_mask.npy".format(os.path.join(data_folder, data_id), data_id))
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).long()
        return X, y

