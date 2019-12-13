import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from data_loaders import Dataset
from models import Modified3DUNet
import os
import torch
import random
from sklearn.model_selection import KFold
from collections import OrderedDict
import pandas as pd
from pandas.plotting import table


##################################################################################################################################
# Code for evaluating the segmentation performance of a model. Returns Dice scores for the whole tumor, the enhancing and the
# necrotic core regions for all the folds as well as the average over the folds. Saves the results in an excel sheet.

# To specify:
preprocessed_data_path = r'/home/artur-cmic/Desktop/UCL/Brats2019/Data/Preprocessed'
model_saves_path = 'pretrained_models'  # Folder where the model saves should be stored in the format Fold_<fold_nr>_Epoch_<epoch_nr>.tar
epoch_nr = 20  # Epoch at which to take the model saves (determined from loss plots)
parallel_training = 1  # Specify if the model was trained with multiple GPUs (slightly more code needed to load model save)
##################################################################################################################################


# Get paths and names (IDS) of folders that store the multimodal training data
folder_paths = []
folder_ids = []
for subdir in os.listdir(preprocessed_data_path):
    folder_paths.append(os.path.join(preprocessed_data_path, subdir))
    folder_ids.append(subdir)

# Shuffle them around, keeping same seed to make sure same shuffling is used if training is interrupted and needs to be continued
random.seed(4)
random.shuffle(folder_paths)
random.seed(4)
random.shuffle(folder_ids)

# Use GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Setup KFold Cross Validation
n_folds = 5  # Number of folds in cross-validation
kf = KFold(n_splits=n_folds, shuffle=False)  # Shuffle=false to get the same shuffling scheme every run

results = pd.DataFrame()
fold_nr = 1
for fold in kf.split(folder_paths):
    print("Getting Model Performance on Fold {}/{}".format(fold_nr, n_folds))
    valid_idx = fold[1]
    valid_set = Dataset([folder_paths[i] for i in valid_idx], [folder_ids[i] for i in valid_idx])
    model = Modified3DUNet(4, 4, 16)
    checkpoint = torch.load("{}/Fold_{}_Epoch_{}.tar".format(model_saves_path,fold_nr, epoch_nr))

    if(parallel_training): # If training was done on multiple GPUs have to rename keys in saved model_state_dict:
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k[7:]  # remove module.
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    fold_measures = {}
    for idx in range(0, len(valid_idx)): # Loop over all the data examples
        with torch.no_grad():
            scans, labels = valid_set[idx]
            scans = np.expand_dims(scans,0)
            scans = torch.from_numpy(scans).to(device)
            output, seg_layer = model(scans)
            seg_layer = seg_layer.squeeze()
            _, indices = seg_layer.max(0)
            indices = indices.cpu().detach().numpy()

            whole_tumor_pred = ((indices == 3) | (indices == 1)).astype('int')
            enhancing_pred = (indices == 3).astype('int')
            core_pred = (indices == 1).astype('int')

            whole_tumor_gt = ((labels == 3) | (labels == 1)).astype('int')
            enhancing_gt = (labels == 3).astype('int')
            core_gt = (labels == 1).astype('int')

            comparisons = {'Whole': {'pred': whole_tumor_pred, 'gt': whole_tumor_gt}, 'Enhancing': {'pred': enhancing_pred, 'gt': enhancing_gt},
                           'Core': {'pred': core_pred, 'gt': core_gt}}
            measures = {}
            for region_name, masks in comparisons.items():
                overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
                overlap_measures_filter.Execute(sitk.GetImageFromArray(masks['pred']), sitk.GetImageFromArray(masks['gt']))

                #dice = overlap_measures_filter.GetDiceCoefficient()
                #plt.figure()
                #plt.subplot(1,2,1)
                #plt.imshow(indices[:,:,50])
                #plt.title('Model output')
                #plt.subplot(1,2,2)
                #plt.imshow(labels[:,:,50])
                #plt.title('Ground truth')
                #plt.suptitle("Dice score = " + str(dice))
                #plt.show()

                measures['{}, Dice'.format(region_name)] = overlap_measures_filter.GetDiceCoefficient()
                #measures['{}, Jaccard'.format(region_name)] = overlap_measures_filter.GetJaccardCoefficient()
            fold_measures[idx] = measures

    fold_measures = pd.DataFrame.from_dict(fold_measures, orient='index').astype('float')
    fold_measures = fold_measures.replace([np.inf, -np.inf], np.nan)
    fold_mean = fold_measures.mean(axis=0, skipna=True)
    fold_std = fold_measures.std(axis=0, skipna=True)
    mean_keys = [x + ', Mean' for x in fold_measures.keys()]
    std_keys = [x + ', Std' for x in fold_measures.keys()]
    if(fold_nr == 1):
        results = pd.DataFrame(columns = mean_keys + std_keys)
        results.loc['Fold {}'.format(fold_nr)] = list(fold_mean) + list(fold_std)
    else:
        results.loc['Fold {}'.format(fold_nr)] = list(fold_mean) + list(fold_std)
    fold_nr += 1

results.loc['Overall'] = results.mean(axis=0)
results = results.astype('float').round(3)
a = results.columns.str.split(', ', expand=True).values
results.columns = pd.MultiIndex.from_tuples([x for x in a])
results = results.sort_index(1)
writer = pd.ExcelWriter('overlap_results.xlsx', engine='xlsxwriter')
results.to_excel(writer)
writer.save()

print("Segmentation Performance: ")
print(results)


