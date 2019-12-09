import os
from radiomics import firstorder, shape, glcm, glszm, glrlm, ngtdm, gldm
import numpy as np
import SimpleITK as sitk
import time
from models import Modified3DUNet
import torch
import pandas as pd

###################################################################
# Code for extracting all imaging features from preprocessed training set and saving them in a csv file, along with the
# age and survival outcome inserted at the last two columns.

# To specify - path where the preprocessed mri scans are stored, path where to load model from for obtaining segmentations
# and path of survival data csv file
mri_data_path = r"/home/artur-cmic/Desktop/UCL/Brats2019/Data/Preprocessed"
model_path = "pretrained_models/Fold_1_Epoch_100.tar"
survival_data_path = r'/home/artur-cmic/Desktop/UCL/Brats2019/Data/MICCAI_BraTS_2019_Data_Training/survival_data.csv'
####################################################################

# Function to extract all the imaging features given folder_path and folder_id of a person
def extract_features(folder_path, folder_id):
    # Load in preprocessed mri volumes
    scans = np.load(r"{}/{}_scans.npy".format(folder_path, folder_id))

    # Get t1ce and flair image from which to extract features
    t1ce_img = sitk.GetImageFromArray(scans[1])
    flair_img = sitk.GetImageFromArray(scans[3])

    # Convert scans from numpy to torch tensor and obtain segmentations with the model. Must Unsqueeze to be in format (B,C,H,W,D)
    scans = torch.unsqueeze(torch.from_numpy(scans),0).to(device)
    _, mask = model(scans)
    mask = torch.squeeze(mask,0)
    _, mask = mask.max(0)
    mask = mask.cpu().detach().numpy()
    enhancing = (mask == 3).astype('long')
    edema = (mask == 2).astype('long')
    ncr_nenhancing = (mask == 1).astype('long')
    whole_tumor = (mask > 0).astype('long')

    regions = {'edema': {'mask': edema, 'modality': flair_img}, 'enhancing': {'mask': enhancing, 'modality': t1ce_img},
               'ncr_nenhancing': {'mask':ncr_nenhancing, 'modality': t1ce_img}, 'whole_tumor': {'mask':whole_tumor, 'modality':t1ce_img}}

    # Convert the region arrays into SITK image objects so they can be inputted to the PyRadiomics featureextractor functions.
    all_features = {}
    printed = 0
    for (region_name, images) in regions.items():

        lbl_img = sitk.GetImageFromArray(images['mask'])
        if len(np.unique(images['mask'])) > 1:

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
            if(not printed):
                #print(folder_id)
                printed = 1
    return all_features


# Get paths and names (IDS) of folders that store the preprocessed data for each example
folder_paths = []
folder_ids = []
for subdir in os.listdir(mri_data_path):
    folder_paths.append(os.path.join(mri_data_path, subdir))
    folder_ids.append(subdir)

# Load Model for getting segmentations with it
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Model Parameters
in_channels = 4
n_classes = 4
base_n_filter = 16

model = Modified3DUNet(in_channels, n_classes, base_n_filter)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

features = {}
start = time.time()
for idx in range(0, len(folder_paths)): # Loop over every person,
    features[folder_ids[idx]] = extract_features(folder_paths[idx], folder_ids[idx])
    print("Extracted features from person {}/{}".format(idx + 1, len(folder_paths)))


elapsed = time.time() - start
hours, rem = divmod(elapsed, 3600)
minutes, seconds = divmod(rem, 60)
print("Extracting Features took {} min {} s".format(minutes, seconds))

features = pd.DataFrame.from_dict(features, orient='index')

surv_data = pd.read_csv(survival_data_path, index_col=0)

# Get indices of rows which to keep in training data - those that have NaN or Alive values for survival should be removed
to_keep = surv_data['Survival'].str.isdigit()
to_keep[to_keep.isnull()] = False
to_keep = to_keep.astype('bool')
to_keep = to_keep.keys()[to_keep.values]  # Keep these data entries

ages = surv_data['Age'][to_keep].astype('float')  # Only get ages of people who to keep in training data
surv = surv_data['Survival'][to_keep].astype('float')

# Convert survival data in days to categories of less than 10month survival, 10<x<15 months and > 15 months
surv = surv/31
surv[surv < 10] = 0
surv[(surv >= 10) & (surv < 15)] = 1
surv[surv >= 15] = 2
surv = surv.astype('int')

features = features.loc[to_keep]
features['Age'] = ages[features.index]
features['Survival'] = surv[features.index]

features.to_csv('features.csv')

print("Saved Features to file")



