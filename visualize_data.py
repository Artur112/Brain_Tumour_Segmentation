import nibabel as nib
from matplotlib import pyplot as plt
from skimage.transform import resize
# from batchviewer import view_batch
import numpy as np
import torch
import torch.nn.functional as F

a = torch.arange(10)
b = torch.arange(10)
print(a.cpu().detach().numpy())
a = a.view(5,2)
b  = b.view(5,2)
print(a.shape)
print(a.cpu().detach().numpy())

c = a*b
num = c.sum()
print(num.shape)
print(num.cpu().detach().numpy())

print(num.sum())
"""
c = b.contiguous()
d = a.contiguous()
# a has "standard layout" (also known as C layout in numpy) descending strides, and no memory gaps (stride(i-1) == size(i)*stride(i))
print (a.shape, a.stride(), a.data_ptr())
# b has same storage as a (data_ptr), but has the strides and sizes swapped around
print (b.shape, b.stride(), b.data_ptr())
# c is in new storage, where it has been arranged in standard layout (which is "contiguous")
print (c.shape, c.stride(), c.data_ptr())
# d is exactly as a, as a was contiguous all along
print (d.shape, d.stride(), d.data_ptr())

"""


#print(torch.arange(0, 5) % 3)

#oneh = F.one_hot(torch.arange(0,5) % 3, num_classes=5)
#print(oneh)
#print(oneh.shape)
#print(oneh.view(-1))
#print(oneh.view(-1).shape)
#x = nib.load(r'C:\Users\artur\Desktop\UCL\Brats2019\Data\MICCAI_BraTS_2019_Data_Training\data\HGG\BraTS19_CBICA_ABB_1\BraTS19_CBICA_ABB_1_seg.nii.gz')
#print(x.get_data_dtype())
#print(x.header)


#img = nib.Nifti1Image(x, np.eye(4))
#nib.save(img, "CBICA_AUC1.nii.gz")
#view_batch(x, width=240, height = 240)
# hello