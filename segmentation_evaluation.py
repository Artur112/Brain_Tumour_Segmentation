import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from data_loaders_old import Dataset
from models import Modified3DUNet
import os
import torch
from enum import Enum

# Get paths and names (IDS) of folders that store the multimodal training data
data_path = r'C:\Users\artur\Desktop\UCL\Brats2019\Data\MICCAI_BraTS_2019_Data_Training\data'
folder_paths = []
folder_IDS = []
for subdir in os.listdir(data_path):
    folder_paths.append(os.path.join(data_path,subdir))
    folder_IDS.append(subdir)

train_set = Dataset(folder_paths, folder_IDS)

model = Modified3DUNet(4, 4, 16)
#checkpoint = torch.load("../KFold_Cross_Validation/Fold_1_Epoch_125.tar")
model.load_state_dict(torch.load("pretrained_models/Modified3DUNet_Epoch_200.pt"))
#model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

for idx in [12,55,142,22]:
    scans, labels = train_set[idx]
    scans = scans.unsqueeze(0)
    output, seg_layer = model(scans)
    seg_layer = seg_layer.squeeze()
    _, indices = seg_layer.max(0)
    labels = labels.numpy()
    #print(np.unique(labels))
    indices = indices.numpy()
    #print(np.unique(indices))
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(sitk.GetImageFromArray(indices), sitk.GetImageFromArray(labels))
    dice = overlap_measures_filter.GetDiceCoefficient()
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(indices[:,:,50])
    plt.title('Model output')
    plt.subplot(1,2,2)
    plt.imshow(labels[:,:,50])
    plt.title('Ground truth')
    plt.suptitle("Dice score = " + str(dice))
    plt.show()

    measures = {}
    measures['dice'] = overlap_measures_filter.GetDiceCoefficient()
    measures['jaccard'] = overlap_measures_filter.GetJaccardCoefficient()
    measures['volume similarity'] = overlap_measures_filter.GetVolumeSimilarity()
    measures['false negatives'] = overlap_measures_filter.GetFalseNegativeError()
    measures['false positives'] = overlap_measures_filter.GetFalsePositiveError()

    for key, item in measures.items():
        print(key, "\t", item)



