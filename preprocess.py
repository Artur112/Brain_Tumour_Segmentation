import os
from skimage.transform import resize
import nibabel as nib
import numpy as np
import sys
from matplotlib import pyplot as plt

##############################################
# Code for preprocessing all of the scans and storing them in numpy arrays. Done so that the preprocessing and image resizing steps
# wouldn't be repeated during training, thereby speeding up the training process. Assumes the data stored in raw_data_path are in two folders: HGG and LGG.

# INPUT:
raw_data_path = r'C:\Users\artur\Desktop\UCL\Brats2019\Data\MICCAI_BraTS_2019_Data_Validation\data'
# raw_data_path = r"/home/artur-cmic/Desktop/UCL/Brats2019/Data/MICCAI_BraTS_2019_Data_Training/data"
save_preprocessed_data_path = r'C:/Users/artur/Desktop/UCL/Brats2019/Data/Preprocessed_Validation'
# save_preprocessed_data_path = r"/home/artur-cmic/Desktop/UCL/Brats2019/Data/Preprocessed"
train_data = False  # Specify whether to preprocess Brats training or valid data
# OUTPUT: Preprocessed data stored in numpy arrays in the save_preprocessed_path
##############################################

# Create folder to store preprocessed data in, exit if folder already exists.
if not os.path.isdir(save_preprocessed_data_path):
    os.mkdir(save_preprocessed_data_path)

# Get folder paths and ids of where the raw scans are stored
folder_paths = []
folder_IDS = []
if train_data:
    for grade in os.listdir(raw_data_path):
        for subdir in os.listdir(os.path.join(raw_data_path, grade)):
            folder_paths.append(os.path.join(raw_data_path, grade, subdir))
            folder_IDS.append(subdir)
else:
    for subdir in os.listdir(raw_data_path):
        folder_paths.append(os.path.join(raw_data_path, subdir))
        folder_IDS.append(subdir)

i = 1
for patient in range(len(folder_paths)):
    data_folder = folder_paths[patient]
    data_id = folder_IDS[patient]
    os.mkdir(os.path.join(save_preprocessed_data_path, data_id))

    output_size = (128,128,128) # Size to change images to

    # Load in and resize the mri images
    img_t1 = resize(nib.load(os.path.join(data_folder, data_id) + "_t1.nii.gz").get_fdata(), output_size)
    img_t1ce = resize(nib.load(os.path.join(data_folder, data_id) + "_t1ce.nii.gz").get_fdata(), output_size)
    img_t2 = resize(nib.load(os.path.join(data_folder, data_id) + "_t2.nii.gz").get_fdata(), output_size)
    img_flair = resize(nib.load(os.path.join(data_folder, data_id) + "_flair.nii.gz").get_fdata(), output_size)

    if train_data:
        # Load in labels
        img_segm = nib.load(os.path.join(data_folder, data_id) + "_seg.nii.gz").get_fdata().astype('long')
        # Segmentation Mask has labels 0,1,2,4. Will change these to 0,1,2,3 and perform resizing on the labels separately
        # Combine them afterwards. Multiplication of labels by 10 so difference between label and background pixel would be
        # greater, otherwise resize wont work properly.
        img_segm[img_segm == 4] = 3
        lbl1 = resize((img_segm == 1)*10, output_size, preserve_range=True, anti_aliasing=True)
        lbl2 = resize((img_segm == 2)*10, output_size, preserve_range=True, anti_aliasing=True)
        lbl3 = resize((img_segm == 3)*10, output_size, preserve_range=True, anti_aliasing=True)

        # Remove uncertain pixels at the edges of a label area before merging labels. Essentially remove pixels with class belonging confidence of < 30%.
        # Made equal to -1 instead of 0, just to make sure that the background pixels are always assigned label 0 with argmax.
        lbl1[lbl1 < 3] = -1
        lbl2[lbl2 < 3] = -1
        lbl3[lbl3 < 3] = -1

        # Merge labels by taking argmax of label values - for a pixel that belongs to two classes after resize, assign to the class
        # that its value is highest for. Adding np.zeros to the first dimension so np.argmax would give 1 for label 1 and not 0.
        img_segm = np.argmax(np.asarray([np.zeros(output_size),lbl1, lbl2, lbl3]),axis=0).astype('long')

    # Preprocess mri volumes
    X = []
    for modality in [img_t1, img_t1ce, img_t2, img_flair]:
        brain_region = modality > 0  # Get region of brain to only manipulate those voxels
        mean = np.mean(modality[brain_region])
        stdev = np.std(modality[brain_region])
        new_img = np.zeros(output_size)
        new_img[brain_region] = (modality[brain_region] - mean) / stdev  # Standardize by mean and stdev
        new_img[new_img > 5] = 5  # Clip outliers
        new_img[new_img < -5] = -5
        Maximum = np.max(new_img)
        Minimum = np.min(new_img[brain_region])
        Range = Maximum - Minimum
        new_img[brain_region] = ((((new_img[brain_region] - Minimum) / Range - 0.5) * 2) + 1) / 2  # Scale to be between 0 and 1
        X.append(new_img.astype('float32'))

    np.save("{}/{}/{}_scans.npy".format(save_preprocessed_data_path, data_id, data_id), X)
    if train_data:
        np.save("{}/{}/{}_mask.npy".format(save_preprocessed_data_path, data_id, data_id), img_segm)
    print("Preprocessed patient {}/{} scans".format(i, len(folder_paths)))
    i = i + 1
