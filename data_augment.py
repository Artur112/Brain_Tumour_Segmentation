import torch
from random import randint
import numpy as np
import random
from torchvision import transforms
import os
from matplotlib import pyplot as plt
#x = torch.arange(128).view(4, 4, 2, 2, 2)
#print(x.shape)

x = np.load(r'C:\Users\artur\Desktop\UCL\Brats2019\Data\preprocessed_training_data_new_2\BraTS19_2013_21_1\BraTS19_2013_21_1_scans.npy')
#plt.imshow(x[1,:,:,40])
#plt.show()
#print(x.shape)
x = torch.from_numpy(x)
# Random rotation
#if(random.random() > 0.5):
rotate_nr = randint(0, 3) #Number of times to rotate
rotate_dir = random.sample([1, 2 , 3], k=2) #Direction of axis in which to rotate
x = torch.rot90(x, rotate_nr, rotate_dir)

# Random flip
#if(random.random() > 0.5):
flip_dir = random.sample([1, 2 , 3], k=2) #Direction of axis in which to flip
x = torch.flip(x,flip_dir)

# Random gamma correction
gamma = random.uniform(0.5, 3)
x = torch.pow(x, gamma)
print(x.shape)
# Random scaling
scale_nr = random.sample([0.75, 1, 1.25], k=1)[0]
dim = x.shape[2]
out_dim = int(dim*scale_nr)
x[0,:,:,:] = torch.re   (x[0,:,:,:], (out_dim, out_dim, out_dim))
print(x.shape)
plt.imshow(x[0,:,:,60],cmap='gray')
plt.show()