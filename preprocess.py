import os
from skimage.transform import resize
import nibabel as nib
import numpy as np
import sys

##############################################
# Code for pre processing all of the scans, storing them in numpy arrays in the format that the 3D UNet model accepts and
# saving all of these to disk. Done such that the preprocessing and image resizing steps wouldn't be repeated during training,
# thereby speeding up the training process. Assumes the data stored in raw_data_path are in two folders: HGG and LGG.

# Specify the following two paths:
#raw_data_path = r'C:\Users\artur\Desktop\UCL\Brats2019\Data\MICCAI_BraTS_2019_Data_Training\data'
raw_data_path = r"/home/artur-cmic/Desktop/Brats2019/Data/Training_Data"
#save_preprocessed_data_path = r'C:\Users\artur\Desktop\UCL\Brats2019\Data\preprocessed_training_data_new'
save_preprocessed_data_path = r"/home/artur-cmic/Desktop/Brats2019/Data/Preprocessed"
##############################################

if not os.path.isdir(save_preprocessed_data_path):
    os.mkdir(save_preprocessed_data_path)
else:
    print("Folder already exists !")
    sys.exit()

folder_paths = []
folder_IDS = []
for grade in os.listdir(raw_data_path):
    for subdir in os.listdir(os.path.join(raw_data_path, grade)):
        folder_paths.append(os.path.join(raw_data_path, grade, subdir))
        folder_IDS.append(subdir)

i = 1
for patient in range(0,len(folder_paths)):
    data_folder = folder_paths[patient]
    data_id = folder_IDS[patient]
    os.mkdir(os.path.join(save_preprocessed_data_path, data_id))

    output_size = (128,128,128) # Size to change images to

    # Load in and resize the volumes
    img_t1 = resize(nib.load(os.path.join(data_folder, data_id) + "_t1.nii.gz").get_fdata(), output_size)
    img_t1ce = resize(nib.load(os.path.join(data_folder, data_id) + "_t1ce.nii.gz").get_fdata(), output_size)
    img_t2 = resize(nib.load(os.path.join(data_folder, data_id) + "_t2.nii.gz").get_fdata(), output_size)
    img_flair = resize(nib.load(os.path.join(data_folder, data_id) + "_flair.nii.gz").get_fdata(), output_size)
    img_segm = resize(nib.load(os.path.join(data_folder, data_id) + "_seg.nii.gz").get_fdata(), output_size)

    # Preprocess
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
        X.append(new_img)

    np.save("{}/{}/{}_scans.npy".format(save_preprocessed_data_path, data_id, data_id), X)
    np.save("{}/{}/{}_mask.npy".format(save_preprocessed_data_path, data_id, data_id), img_segm.astype('int'))
    print("Preprocessed patient {}/{} scans".format(i, len(folder_paths)))
    i = i + 1
