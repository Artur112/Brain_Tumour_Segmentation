import numpy as np
import nibabel as nib
from data_loaders import Dataset
from models import Modified3DUNet
import os
import torch
from collections import OrderedDict

preprocessed_valid_data_path = r'C:/users/artur/Desktop/UCL/Brats2019/Data/Preprocessed_Validation'
model_saves_path = r'C:/users/artur/Desktop/UCL/Brats2019/Model_Saves/V3_Prepro_Aug_CrossEn'  # Folder where the model saves should be stored in the format Fold_<fold_nr>_Epoch_<epoch_nr>.tar
epoch_nr = 180  # Epoch at which to take the model saves (determined from loss plots)
parallel_training = 0  # Specify if the model was trained with multiple GPUs (slightly more code needed to load model save)
save_results_path = r'C:/users/artur/Desktop/UCL/Brats2019/Seg_Results'

# Get paths and names (IDS) of folders that store the multimodal training data
folder_paths = []
folder_ids = []
for subdir in os.listdir(preprocessed_valid_data_path):
    folder_paths.append(os.path.join(preprocessed_valid_data_path, subdir))
    folder_ids.append(subdir)

valid_set = Dataset(folder_paths, folder_ids, False)

# Use GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

model = Modified3DUNet(4, 4, 16)
checkpoint = torch.load("{}/Fold_{}_Epoch_{}.tar".format(model_saves_path, 1, epoch_nr))

if parallel_training:  # If training was done on multiple GPUs have to rename keys in saved model_state_dict:
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        name = k[7:]  # remove module.
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
else:
    model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

for idx in range(0, len(folder_ids)):
    with torch.no_grad():
        scans = valid_set[idx]
        scans = np.expand_dims(scans, 0)
        scans = torch.from_numpy(scans).to(device)
        output, seg_layer = model(scans)
        seg_layer = seg_layer.squeeze()
        _, indices = seg_layer.max(0)
        indices = indices.cpu().detach().numpy()
        indices[indices == 3] = 4
        img = nib.Nifti1Image(indices, np.eye(4))
        nib.save(img, "{}/{}.nii.gz".format(save_results_path, folder_ids[idx]))
        print('Saved example {}/{}'.format(idx + 1, len(folder_ids)))