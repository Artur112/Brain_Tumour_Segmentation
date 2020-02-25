import os
import nibabel as nib
import numpy as np
import time
import shutil
import pandas
from matplotlib import pyplot as plt
import sys
import json
"""
path = r"/home/ajurgens/Brats2019/MIDL/brats_reg"
iii = 0
for subdir in os.listdir(path):
    print("Clipping {}/{}".format(iii,len(os.listdir(path))))
    ii = 0
    for aug in os.listdir(os.path.join(path,subdir)):
        if(ii>30):
            shutil.rmtree(os.path.join(path,subdir,aug))
        ii = ii + 1
    iii += 1

data = pandas.read_csv('/home/artur-cmic/Downloads/runusers/artur/Desktop/UCL/run-testing-tag-Valid_Loss_per_Epoch.csv')

steps = list(data['Step'])
values = list(data['Value'])
plt.plot(steps, values)
plt.show()
weight = 0.95
last = values[0]
smoothed = list()
for point in values:
    smoothed_val = last*weight + (1-weight)*point
    smoothed.append(smoothed_val)
    last = smoothed_val

plt.plot(steps, smoothed)
plt.show()

aug = []
if len(sys.argv) > 1:
    for m in range(1,len(sys.argv)):
        aug.append(sys.argv[m])
print(aug)

# Get last epoch nr
epoch_nrs = []
for model_save in os.listdir("/home/artur-cmic/Desktop/UCL/Data_Aug_Experiments/Model_Saves/Baseline_10_1"):
    epoch_nr = model_save[model_save.find("_") + 1:-4]
    epoch_nrs.append(int(epoch_nr))

epoch_nr = max(epoch_nrs)
print(epoch_nr)

import pandas as pd

data = pd.read_excel('Database Logos and Cancel Links.xlsx', header=0)
data_json = {}

for i in range(len(data)):
    info = {}
    name = data.iloc[i]['Name']
    name = name.replace(" ", "").replace("'","").replace("`","").lower()
    info['Logo Link'] = data.iloc[i]['Logo Link']
    info['Cancel Link'] = data.iloc[i]['Cancel Link']
    data_json[name] = info

with open('company_links.json', 'w') as fp:
    json.dump(data_json, fp, indent=3)
"""
#####################################
start_time = time.time()
data_folder = '/home/artur-cmic/Desktop/UCL/Data_Aug_Experiments/Data/Training_Data/Raw/BraTS19_CBICA_ALU_1'
data_id = 'BraTS19_CBICA_ALU_1'

# Load in and resize the mri images
img_t1 = nib.load(os.path.join(data_folder, data_id) + "_t1.nii.gz").get_fdata()
img_t1ce = nib.load(os.path.join(data_folder, data_id) + "_t1ce.nii.gz").get_fdata()
img_t2 = nib.load(os.path.join(data_folder, data_id) + "_t2.nii.gz").get_fdata()
img_flair = nib.load(os.path.join(data_folder, data_id) + "_flair.nii.gz").get_fdata()

# Load in labels
img_segm = nib.load(os.path.join(data_folder, data_id) + "_seg.nii.gz").get_fdata().astype('long')
# Replace label 4 with label 3
img_segm[img_segm == 4] = 3

# Preprocess mri volumes
X = []
for modality in [img_t1, img_t1ce, img_t2, img_flair]:
    brain_region = modality > 0  # Get region of brain to only manipulate those voxels
    mean = np.mean(modality[brain_region])
    stdev = np.std(modality[brain_region])
    new_img = np.zeros(img_t1.shape)
    new_img[brain_region] = (modality[brain_region] - mean) / stdev  # Standardize by mean and stdev
    #new_img[new_img > 5] = 5  # Clip outliers
    #new_img[new_img < -5] = -5
    Maximum = np.max(new_img)
    Minimum = np.min(new_img[brain_region])
    Range = Maximum - Minimum
    new_img[brain_region] = (new_img[brain_region] - Minimum) / Range  # Scale to be between 0 and 1
    X.append(new_img.astype('float32'))

elapsed_time = time.time() - start_time
print(elapsed_time)
print('Time_{}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))


start_time = time.time()
a = np.load('savez_compressed_scans.npz')['arr_0']
b = np.load('savez__compressed_mask.npz')['arr_0']
print(b.shape)
elapsed_time = time.time() - start_time
print(elapsed_time)
print('Time_{}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))




