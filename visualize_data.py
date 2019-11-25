import nibabel as nib
from matplotlib import pyplot as plt
from skimage.transform import resize

x = resize(nib.load(r'C:\Users\artur\Desktop\UCL\Brats2019\Data\MICCAI_BraTS_2019_Data_Training\data\HGG\BraTS19_CBICA_AAG_1\BraTS19_CBICA_AAG_1_t1.nii.gz').get_fdata(), (128, 128, 128))
#x = nib.load(r'C:\Users\artur\Desktop\UCL\Brats2019\Data\MICCAI_BraTS_2019_Data_Training\data\HGG\BraTS19_CBICA_AAG_1\BraTS19_CBICA_AAG_1_t1.nii.gz').get_fdata()
print(x.shape)
plt.imshow(x[60,:,:])
plt.show()