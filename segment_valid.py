import numpy as np
import nibabel as nib
from data_loaders import Dataset
from models import Modified3DUNet
from skimage.transform import resize
import os
import torch
from collections import OrderedDict

#preprocessed_valid_data_path = r'C:/users/artur/Desktop/UCL/Brats2019/Data/Preprocessed_Validation'
preprocessed_valid_data_path = r'/home/artur-cmic/Desktop/UCL/Brats2019/Data/Preprocessed_Validation'
#model_saves_path = r'C:/users/artur/Desktop/UCL/Brats2019/Model_Saves/V4_Prepro_NoAug_Dice'  # Folder where the model saves should be stored in the format Fold_<fold_nr>_Epoch_<epoch_nr>.tar
model_saves_path = r'/home/artur-cmic/Desktop/UCL/Brats2019/Model_Saves/V4_Prepro_NoAug_Dice'
epoch_nr = 250  # Epoch at which to take the model saves (determined from loss plots)
parallel_training = 1  # Specify if the model was trained with multiple GPUs (slightly more code needed to load model save)
#save_results_path = r'C:/users/artur/Desktop/UCL/Brats2019/Seg_Results_V4'
save_results_path = r'/home/artur-cmic/Desktop/UCL/Brats2019/Seg_Results/Seg_Results_V4'
folds = [1, 2, 3, 4, 5]  # Fold across which to ensemble average output

if not os.path.isdir(save_results_path):
    os.mkdir(save_results_path)

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

models = []
for fold in folds:
    model = Modified3DUNet(4, 4, 16)
    checkpoint = torch.load("{}/Fold_{}_Epoch_{}.tar".format(model_saves_path, fold, epoch_nr))
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
    models.append(model)

for idx in range(0, len(folder_ids)):
    with torch.no_grad():
        scans = valid_set[idx]
        scans = np.expand_dims(scans, 0)
        scans = torch.from_numpy(scans).to(device)
        seg_results = []
        for fold in folds:
            output, seg_layer = models[fold-1](scans)
            seg_layer = seg_layer.squeeze()
            seg_results.append(seg_layer)

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
        print('Saved example {}/{}'.format(idx + 1, len(folder_ids)))