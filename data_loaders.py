import numpy as np
from torch.utils import data


##########################################################################
# Dataset class that feeds data into a data generator. Takes in a list of folder paths in which a patient's scans are stored,
# a list of the names of the folder / patient ids and whether segmentation labels are provided and should be loaded.
# Returns two torch arrays, one that contains the multi-modal scans and the other that contains the masks.
##########################################################################


class Dataset(data.Dataset):
    def __init__(self, folder_path, folder_id, seg_provided = True):
        self.folder_paths = folder_path
        self.folder_ids = folder_id
        self.seg_provided = seg_provided

    def __len__(self):
        return len(self.folder_ids)

    def __getitem__(self, index):
        data_folder = self.folder_paths[index]
        data_id = self.folder_ids[index]
        X = np.load(r"{}/{}_scans.npy".format(data_folder, data_id))
        if self.seg_provided:
            y = np.load(r"{}/{}_mask.npy".format(data_folder, data_id))
            return X, y
        else:
            return X

