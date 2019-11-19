import os
from radiomics import firstorder, shape, glcm, glszm, glrlm, ngtdm, gldm
import numpy as np
import SimpleITK as sitk


# Function to extract all the imaging features given folder_path and folder_id
def extract_all_features(folder_path, folder_ID):
    all_features = {}

    # Load the images - t1ce and flair only used
    t1ce_img = sitk.ReadImage(folder_path + "\\" + folder_ID + "_t1ce.nii.gz")
    flair_img = sitk.ReadImage(folder_path + "\\" + folder_ID + "_flair.nii.gz")
    labels_img = sitk.ReadImage(folder_path + "\\" + folder_ID + "_seg.nii.gz")

    # Convert the label image to to a numpy array and then separate it into the different regions/labels
    labels = sitk.GetArrayFromImage(labels_img)
    edema = (labels == 2).astype(np.int16)
    enhancing = (labels == 4).astype(np.int16)
    ncr_nenhancing = (labels == 1).astype(np.int16)
    #whole_tumor = (labels > 0).astype(np.int16)

    regions = {'edema': {'mask': edema, 'modality': flair_img}, 'enhancing': {'mask': enhancing, 'modality': t1ce_img}, 'ncr_nenhancing': {'mask':ncr_nenhancing, 'modality': t1ce_img}}
    # Convert the region arrays into SITK image objects so they can be inputted to the PyRadiomics featureextractor functions.
    # Set the origin of them to be the same as that of the label_img, as converting an SITK image to a numpy array and back changes the origin of the matrix.
    name = 1
    for (region_name, images) in regions.items():
        lbl_img = sitk.GetImageFromArray(images['mask'])
        lbl_img.SetOrigin(labels_img.GetOrigin())

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

    return all_features

# Path where to load the data from
data_path = r'C:\Users\artur\Desktop\UCL\Brats2019\Data\MICCAI_BraTS_2019_Data_Training\data'

# Get paths and names (IDS) of folders that store the multimodal training data for each example
folder_paths = []
folder_IDS = []
for grade in os.listdir(data_path):
    for subdir in os.listdir(os.path.join(data_path, grade)):
        folder_paths.append(os.path.join(data_path, grade, subdir))
        folder_IDS.append(subdir)

for idx in range(0,1): # Loop over every image
    features = extract_all_features(folder_paths[idx], folder_IDS[idx])
    for (key, val) in features.items():
        print("\t%s: %s" % (key, val))
