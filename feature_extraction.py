import os
from radiomics import firstorder, shape, glcm, glszm, glrlm, ngtdm, gldm
import numpy as np
import SimpleITK as sitk
import time
from skimage.transform import resize
import nibabel as nib
from models import Modified3DUNet
import torch
import json

# Function to extract all the imaging features given folder_path and folder_id of a person
def extract_all_features(data_folder, data_id, model, device):

    all_features = {}
    output_size = (128, 128, 128)  # Size to change images to

    # Load in and resize the volumes
    img_t1 = resize(nib.load(os.path.join(data_folder, data_id) + "_t1.nii.gz").get_fdata(), output_size)
    img_t1ce = resize(nib.load(os.path.join(data_folder, data_id) + "_t1ce.nii.gz").get_fdata(), output_size)
    img_t2 = resize(nib.load(os.path.join(data_folder, data_id) + "_t2.nii.gz").get_fdata(), output_size)
    img_flair = resize(nib.load(os.path.join(data_folder, data_id) + "_flair.nii.gz").get_fdata(), output_size)

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
        X.append(new_img.astype('float32'))

    scans = torch.from_numpy(np.asarray(X))
    scans = torch.unsqueeze(scans,0).to(device)

    # Get segmentation map
    #output, seg_layer = model(scans)

    # Load the MRI images from which to derive features - t1ce and flair only used
    t1ce_img = sitk.GetImageFromArray(X[1])
    flair_img = sitk.GetImageFromArray(X[3])

    img_segm = nib.load(os.path.join(data_folder, data_id) + "_seg.nii.gz").get_fdata().astype('long')

    # Segmentation Mask has labels 0,1,2,4. Will change these to 0,1,2,3. Perform resizing on the labels separately
    # and then combine them afterwards.
    img_segm[img_segm == 4] = 3
    lbl1 = resize((img_segm == 1) * 10, output_size, preserve_range=True, anti_aliasing=True)
    lbl2 = resize((img_segm == 2) * 10, output_size, preserve_range=True, anti_aliasing=True)
    lbl3 = resize((img_segm == 3) * 10, output_size, preserve_range=True, anti_aliasing=True)
    labels = np.zeros(output_size).astype('long')
    labels[lbl1 > 1] = 1
    labels[lbl2 > 1] = 2
    labels[lbl3 > 1] = 3

    # Convert the label image to to a numpy array and then separate it into the different regions/labels
    #seg_layer = torch.squeeze(seg_layer,0)
    #_, labels = seg_layer.max(0)
    #labels = labels.detach().cpu().numpy()
    #nib.save(nib.Nifti1Image(labels,affine=np.eye(4)), '/home/artur-cmic/Desktop/labels.nii')

    ncr_nenhancing = (labels == 1).astype(np.int16)
    edema = (labels == 2).astype(np.int16)
    enhancing = (labels == 3).astype(np.int16)

    regions = {'edema': {'mask': edema, 'modality': flair_img}, 'enhancing': {'mask': enhancing, 'modality': t1ce_img}, 'ncr_nenhancing': {'mask':ncr_nenhancing, 'modality': t1ce_img}}
    # Convert the region arrays into SITK image objects so they can be inputted to the PyRadiomics featureextractor functions.
    name = 1
    for (region_name, images) in regions.items():

        lbl_img = sitk.GetImageFromArray(images['mask'])

        if(len(np.unique(images['mask'])) > 1):
            # Get First order features
            firstorderfeatures = firstorder.RadiomicsFirstOrder(images['modality'], lbl_img)
            firstorderfeatures.enableAllFeatures()  # On the feature class level, all features are disabled by default
            firstorderfeatures.execute()
            for (key, val) in firstorderfeatures.featureValues.items():
                all_features[region_name + '_' + key] = val

            # Get Shape features
            shapefeatures = shape.RadiomicsShape(images['modality'], lbl_img)
            shapefeatures.enableAllFeatures()
            shapefeatures.execute()
            for (key, val) in shapefeatures.featureValues.items():
                all_features[region_name + '_' + key] = val

            # Get Gray Level Co-occurrence Matrix (GLCM) Features
            glcmfeatures = glcm.RadiomicsGLCM(images['modality'], lbl_img)
            glcmfeatures.enableAllFeatures()
            glcmfeatures.execute()
            for (key, val) in glcmfeatures.featureValues.items():
                all_features[region_name + '_' + key] = val

            # Get Gray Level Size Zone Matrix (GLSZM) Features
            glszmfeatures = glszm.RadiomicsGLSZM(images['modality'], lbl_img)
            glszmfeatures.enableAllFeatures()
            glszmfeatures.execute()
            for (key, val) in glszmfeatures.featureValues.items():
                all_features[region_name + '_' + key] = val

            # Get Gray Level Run Length Matrix (GLRLM) Features
            glrlmfeatures = glrlm.RadiomicsGLRLM(images['modality'], lbl_img)
            glrlmfeatures.enableAllFeatures()
            glrlmfeatures.execute()
            for (key, val) in glrlmfeatures.featureValues.items():
                all_features[region_name + '_' + key] = val

            # Get Neighbouring Gray Tone Difference Matrix (NGTDM) Features
            ngtdmfeatures = ngtdm.RadiomicsNGTDM(images['modality'], lbl_img)
            ngtdmfeatures.enableAllFeatures()
            ngtdmfeatures.execute()
            for (key, val) in ngtdmfeatures.featureValues.items():
                all_features[region_name + '_' + key] = val

            # Get Gray Level Dependence Matrix (GLDM) Features
            gldmfeatures = gldm.RadiomicsGLDM(images['modality'], lbl_img)
            gldmfeatures.enableAllFeatures()
            gldmfeatures.execute()
            for (key, val) in gldmfeatures.featureValues.items():
                all_features[region_name + '_' + key] = val
        else:
            print(data_id)
    return all_features


# Path where to load the data from
data_path = r"/home/artur-cmic/Desktop/Brats2019/Data/Training_Data"

# Get paths and names (IDS) of folders that store the multimodal training data for each example
folder_paths = []
folder_IDS = []
for grade in os.listdir(data_path):
    for subdir in os.listdir(os.path.join(data_path, grade)):
        folder_paths.append(os.path.join(data_path, grade, subdir))
        folder_IDS.append(subdir)

# Load Model for getting segmentations with it
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
# Model Parameters
in_channels = 4
n_classes = 4
base_n_filter = 16

model = Modified3DUNet(in_channels, n_classes, base_n_filter)
checkpoint = torch.load("pretrained_models/V2_Fold_1_Epoch_143.tar")
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()


features = {}
start = time.time()
for idx in range(0, len(folder_paths)): # Loop over every person,
    features[folder_IDS[idx]] = extract_all_features(folder_paths[idx], folder_IDS[idx], model,device)

elapsed = time.time() - start
hours, rem = divmod(elapsed, 3600)
minutes, seconds = divmod(rem, 60)
print("Extracting Features took {} min {} s".format(minutes,seconds))
with open('features.json', 'w') as fp:
    json.dump(features, fp)

for (key, val) in features.items():
    print("\t%s: %s" % (key, val))
