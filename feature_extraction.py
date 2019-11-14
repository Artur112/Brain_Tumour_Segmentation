import os
from radiomics import featureextractor
import numpy as np
import SimpleITK as sitk

# Path where to load the data from
data_path = r'C:\Users\artur\Desktop\UCL\Brats2019\Data\MICCAI_BraTS_2019_Data_Training\data'

# Get paths and names (IDS) of folders that store the multimodal training data for each example
folder_paths = []
folder_IDS = []
for grade in os.listdir(data_path):
    for subdir in os.listdir(os.path.join(data_path, grade)):
        folder_paths.append(os.path.join(data_path, grade, subdir))
        folder_IDS.append(subdir)

for idx in range(0,1):

    # Load the images
    t1ce_img = sitk.ReadImage(folder_paths[idx] + "\\" + folder_IDS[idx] + "_t1ce.nii.gz")
    flair_img = sitk.ReadImage(folder_paths[idx] + "\\" + folder_IDS[idx] + "_flair.nii.gz")
    labels_img = sitk.ReadImage(folder_paths[idx] + "\\" + folder_IDS[idx] + "_seg.nii.gz")

    # Convert the label image to to a numpy array and then separate it into the different regions
    labels = sitk.GetArrayFromImage(labels_img)
    edema = (labels == 2).astype(np.int16)
    enhancing = (labels == 4).astype(np.int16)
    ncr_nenhancing = (labels == 1).astype(np.int16)
    whole_tumor = (labels > 0).astype(np.int16)

    # Convert the region arrays into SITK image objects so they can be inputted to the PyRadiomics featureextractor function.
    # Set the origin of them to be the same as that of the label_img, as converting an SITK image to a numpy array and back changes the origin.
    edema_img = sitk.GetImageFromArray(edema)
    edema_img.SetOrigin(labels_img.GetOrigin())
    enhancing_img = sitk.GetImageFromArray(edema)
    enhancing_img.SetOrigin(labels_img.GetOrigin())
    ncr_nenhancing_img = sitk.GetImageFromArray(edema)
    ncr_nenhancing_img.SetOrigin(labels_img.GetOrigin())
    whole_tumor_img = sitk.GetImageFromArray(edema)
    whole_tumor_img.SetOrigin(labels_img.GetOrigin())

    # Extract the features
    extractor = featureextractor.RadiomicsFeatureExtractor()
    # result1 = extractor.execute(flair_path, r'Regions\edema.nii', voxelBased=True)
    result1 = extractor.execute(flair_img, edema_img, voxelBased=True)
    # result2 = extractor.execute(t1ce_img, enhancing_img', voxelBased=True)
    # result3 = extractor.execute(t1ce_img, ncr_nenhancing_img, voxelBased=True)

    print("Calculated features")
    for key, value in result1.items():
        print("\t", key, ":", value)
    # print("Calculated features")
    # for key, value in result2.items():
    #    print(r"\t", key, ":", value)
    # print("Calculated features")
    # for key, value in result3.items():
    #    print("\t", key, ":", value)
