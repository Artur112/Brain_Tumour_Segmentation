import os
import numpy as np
import nibabel as nib

raw_data_path = r'C:\Users\artur\Desktop\UCL\Brats2019\Data\MICCAI_BraTS_2019_Data_Training\data'

folder_paths = []
folder_IDS = []
for grade in os.listdir(raw_data_path):
    for subdir in os.listdir(os.path.join(raw_data_path, grade)):
        folder_paths.append(os.path.join(raw_data_path, grade, subdir))
        folder_IDS.append(subdir)

mean_img = np.zeros((240,240,155)).astype('float')
for idx in range(len(folder_IDS)):
    img = nib.load(os.path.join(folder_paths[idx], folder_IDS[idx]) + "_t1.nii.gz").get_fdata()
    mean_img = mean_img + img
    print("{}/{}".format(idx+1,len(folder_IDS)))

mean_img = mean_img / len(folder_IDS)
brats_Transform_matrix = [[-1,-0,-0,0],[-0,-1,-0,239],[0,0,1,0],[0,0,0,1]]
img = nib.Nifti1Image(mean_img, brats_Transform_matrix)
nib.save(img, 'mean_img_Brats.nii.gz')