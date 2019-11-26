import torch
from random import randint
import numpy as np
import random
from torchvision import transforms
import os
from matplotlib import pyplot as plt
import elasticdeform
#x = torch.arange(128).view(4, 4, 2, 2, 2)
#print(x.shape)

x = np.load(r'C:\Users\artur\Desktop\UCL\Brats2019\Data\preprocessed_training_data_new\BraTS19_CBICA_AAL_1\BraTS19_CBICA_AAL_1_scans.npy')
y = np.load(r'C:\Users\artur\Desktop\UCL\Brats2019\Data\preprocessed_training_data_new\BraTS19_CBICA_AAL_1\BraTS19_CBICA_AAL_1_mask.npy')

# Random elastic deformation
if(random.random() > 0.001):
    y = np.unsqueeze(0)
    sigma_nr = random.uniform(0,2.5)
    [x,y] = elasticdeform.deform_random_grid([x, y], axis=(1, 2, 3), sigma=sigma_nr, points=3) # Elastic deformation on iamges and brain mask
    plt.show()
    mask = mask > 0.5
    x = x*mask

x = torch.from_numpy(x)

# Random rotation
if(random.random() > 0.5):
    rotate_nr = randint(0, 3) #Number of times to rotate
    rotate_dir = random.sample([1, 2 , 3], k=2) #Direction of axis in which to rotate
    x = torch.rot90(x, rotate_nr, rotate_dir)

# Random flip
if(random.random() > 0.5):
    flip_dir = random.sample([1, 2 , 3], k=2) # Random axis direction in which to flip
    x = torch.flip(x,flip_dir)

# Random gamma correction
if(random.random() > 0.5):
    gamma = random.uniform(0.5, 3) # Random gamma value by which to correct
    x = torch.pow(x, gamma)

# Random scaling
if(random.random() > 0.99):
    scale_nr = random.sample([0.50, 0.75, 1, 1.25, 1.5], k=1)[0] # Random scale by which to change image size
    dim = x.shape[2]
    out_dim = int(dim*scale_nr)
    x = torch.nn.functional.interpolate(x, (out_dim,out_dim,out_dim), mode='trilinear')
plt.imshow(x[0, :, :, 50], cmap='gray')
plt.show()

