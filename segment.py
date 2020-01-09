import numpy as np
import nibabel as nib
from data_loaders import Dataset
from models import UNet3D
from skimage.transform import resize
import os
import torch
from collections import OrderedDict
from sys import argv

########################################################################################################################
# Code to segment a set of scans after model has been trained. Segmentation obtained by ensemble averaging outputs of the
# models from the different folds. When calling script:

# INPUT arguments:
#   arg1: path to where the preprocessed scans to segment are stored
#   arg2: path to where the Model_Saves are stored
#   arg3: epoch nr at which to to load a models save
#   arg4: path where to save the segmented scans. Code creates folder if it doesnt exist

# OUTPUT:
#   segmented scans stored as nii.gz files in provided save results path
########################################################################################################################

preprocessed_data_path = argv[1]
model_saves_path = argv[2]
epoch_nr = int(argv[3])  # Epoch at which to take the model saves (determined from loss plots)
save_results_path = argv[4]

run_name = model_saves_path[model_saves_path.rindex("/") + 1:]

if not os.path.isdir(save_results_path):
    os.mkdir(save_results_path)

# Get paths and names (IDS) of folders that store the multimodal training data
folder_paths = []
folder_ids = []
for subdir in os.listdir(preprocessed_data_path):
    folder_paths.append(os.path.join(preprocessed_data_path, subdir))
    folder_ids.append(subdir)

valid_set = Dataset(folder_paths, folder_ids, False)

# Use GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

models = []
for fold in range(1,6):
    model = UNet3D(4, 4, False, 16, 'crg', 8)
    checkpoint = torch.load("{}/Fold_{}_Epoch_{}.tar".format(model_saves_path, fold, epoch_nr))
    if list(checkpoint['model_state_dict'].keys())[0].find("module") > -1:
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k[7:]  # remove module.
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    models.append(model)

for idx in range(0, len(folder_ids)):
    with torch.no_grad():
        scans = valid_set[idx]
        scans = np.expand_dims(scans, 0)
        scans = torch.from_numpy(scans).to(device)
        seg_results = []
        for fold in range(1,6):
            output = models[fold-1](scans)
            output = output.squeeze()
            seg_results.append(output)

        seg_layer_avg = torch.mean(torch.stack(seg_results), dim=0)
        #print(seg_layer_avg.shape)
        _, indices = seg_layer_avg.max(0)
        indices = indices.cpu().detach().numpy()

        lbl1 = resize((indices == 1) * 10, (240, 240, 155), preserve_range=True, anti_aliasing=True)
        lbl2 = resize((indices == 2) * 10, (240, 240, 155), preserve_range=True, anti_aliasing=True)
        lbl3 = resize((indices == 3) * 10, (240, 240, 155), preserve_range=True, anti_aliasing=True)

        # Remove uncertain pixels at the edges of a label area before merging labels. Essentially remove pixels with class belonging confidence of < 30%.
        # Made equal to -1 instead of 0, just to make sure that the background pixels are always assigned label 0 with argmax.
        lbl1[lbl1 < 3] = -1
        lbl2[lbl2 < 3] = -1
        lbl3[lbl3 < 3] = -1

        # Merge labels by taking argmax of label values - for a pixel that belongs to two classes after resize, assign to the class
        # that its value is highest for. Adding np.zeros to the first dimension so np.argmax would give 1 for label 1 and not 0.
        img_segm = np.argmax(np.asarray([np.zeros((240, 240, 155)), lbl1, lbl2, lbl3]), axis=0).astype(np.uint8)

        img_segm[img_segm == 3] = 4
        img = nib.Nifti1Image(img_segm, [[-1,-0,-0,0],[-0,-1,-0,239],[0,0,1,0],[0,0,0,1]])
        nib.save(img, "{}/{}.nii.gz".format(save_results_path, folder_ids[idx]))
        print('Saved example {}/{} for run {}'.format(idx + 1, len(folder_ids), run_name))