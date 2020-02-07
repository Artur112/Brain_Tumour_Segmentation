import numpy as np
from torch.utils import data


##########################################################################
# Dataset class that feeds data into a data generator.
# INPUT:
#   folder_path: list of full folder paths of the preprocessed data.
#   folder_id: list of names of those folders.
#   seg_provided: specify False if performing inference and only scans should be loaded.
#
# OUTPUT:
#   X, y numpy arrays, where X = [C,H,W,D] contains C modality scans and y = [H,W,D] contains the segmentation labels.
##########################################################################


class Dataset(data.Dataset):
    def __init__(self, folder_path, folder_id, seg_provided=True):
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

