import os
from skimage.transform import resize
import nibabel as nib
import numpy as np
import sys

##############################################
# Code for preprocessing all of the scans and storing them in numpy arrays. Done so that preprocessing
# wouldn't be repeated during training, thereby speeding up the training process. Assumes the data is all in the specified folder and not
# within sub folders in it (ie "Brats_ID folder thats stores scans for that Brats_ID is in that folder and not a subfolder).

# INPUT arguments:
#   arg1: path to where the raw scans to preprocess are stored
#   arg2: path where to save the preprocessed scans
#   arg3: Specify 1 if to preprocess training data (segmentation labels as well) or 0 if test data only.
#
# OUTPUT:
#   Preprocessed data stored in compressed npz files in the save_preprocessed_path
##############################################

raw_data_path = sys.argv[1]
save_preprocessed_data_path = sys.argv[2]
train_data = int(sys.argv[3])

# Create folder to store preprocessed data in, exit if folder already exists.
if not os.path.isdir(save_preprocessed_data_path):
    os.mkdir(save_preprocessed_data_path)
else:
    print("Folder to store preprocessed images in already exists")
    sys.exit()

# Get folder paths and ids of where the raw scans are stored
folder_paths = []
folder_IDS = []
for subdir in os.listdir(raw_data_path):
    folder_paths.append(os.path.join(raw_data_path, subdir))
    folder_IDS.append(subdir)

i = 1
for patient in range(len(folder_paths)):
    data_folder = folder_paths[patient]
    data_id = folder_IDS[patient]
    os.mkdir(os.path.join(save_preprocessed_data_path, data_id))


    # Load in and resize the mri images
    img_t1 = nib.load(os.path.join(data_folder, data_id) + "_t1.nii.gz").get_fdata()
    img_t1ce = nib.load(os.path.join(data_folder, data_id) + "_t1ce.nii.gz").get_fdata()
    img_t2 = nib.load(os.path.join(data_folder, data_id) + "_t2.nii.gz").get_fdata()
    img_flair = nib.load(os.path.join(data_folder, data_id) + "_flair.nii.gz").get_fdata()

    if train_data:
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

    np.savez_compressed("{}/{}/{}_scans".format(save_preprocessed_data_path,data_id, data_id), X)
    if train_data:
        np.savez_compressed("{}/{}/{}_mask".format(save_preprocessed_data_path, data_id, data_id), img_segm)
    print("Preprocessed patient {}/{} scans".format(i, len(folder_paths)))
    i = i + 1


