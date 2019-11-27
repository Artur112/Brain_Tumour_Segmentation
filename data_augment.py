import torch
from random import randint
import numpy as np
import random
from matplotlib import pyplot as plt
import elasticdeform

def plot(x,y,dim1, dim2, i, dif, titlee):
    slc = 80
    if(x.shape[2] != 128):
        slc = int(((80/128)*x.shape[2]))
    plt.subplot(dim1,dim2,i)
    plt.title(titlee)
    plt.imshow(x[0,:,:,slc],cmap='gray')
    plt.subplot(dim1,dim2,i+dif)
    plt.imshow(y[:,:,slc])


#x = np.load(r'/home/artur-cmic/Desktop/Brats2019/Data/Preprocessed/BraTS19_CBICA_AAL_1/BraTS19_CBICA_AAL_1_scans.npy')
#y = np.load(r'/home/artur-cmic/Desktop/Brats2019/Data/Preprocessed/BraTS19_CBICA_AAL_1/BraTS19_CBICA_AAL_1_mask.npy')
x = np.load(r'C:/users/artur/Desktop/UCL/Brats2019/Data/preprocessed_training_data_new/BraTS19_CBICA_AAL_1/BraTS19_CBICA_AAL_1_scans.npy')
y = np.load(r'C:/users/artur/Desktop/UCL/Brats2019/Data/preprocessed_training_data_new/BraTS19_CBICA_AAL_1/BraTS19_CBICA_AAL_1_mask.npy')
plot(x,y,2,6,1,6,'Original')

# Random elastic deformation
if(random.random() > 0.001):
    brain_region1 = np.asarray(x > 0).astype('float')
    brain_region = brain_region1
    y = np.expand_dims(y,axis=0).astype('float')
    sigma_nr = random.uniform(0,2.5)
    [x,y,brain_region] = elasticdeform.deform_random_grid([x, y,brain_region], axis=(1, 2, 3), sigma=sigma_nr, points=3) # Elastic deformation on iamges and brain mask
    y = np.squeeze(y,0).astype('int')
    x = x*brain_region
plot(x,y,2,6,2,6,'Elastic Deformation')


x = torch.from_numpy(x)
y = torch.from_numpy(y)

# Random rotation
if(random.random() > 0.01):
    rotate_nr = randint(0, 3) #Number of times to rotate
    rotate_dir = random.sample([1, 2 , 3], k=2) #Direction of axis in which to rotate
    x = torch.rot90(x, rotate_nr, rotate_dir)
    y = torch.unsqueeze(y,0)
    y = torch.rot90(y, rotate_nr, rotate_dir)

plot(x,y[0,:],2,6,3,6,'Rotation')

# Random flip
if(random.random() > 0):
    flip_dir = random.sample([1, 2 , 3], k=2) # Random axis direction in which to flip
    x = torch.flip(x,flip_dir)
    y = torch.flip(y,flip_dir)

plot(x,y[0,:],2,6,4,6,'Flip')

# Random gamma correction
if(random.random() > 0):
    gamma = 1.5#random.uniform(0.5, 2) # Random gamma value by which to correct
    x = torch.pow(x, gamma)
plot(x,y[0,:],2,6,5,6,'Gamma Correction')

# Random scaling
if(random.random() > 0):
    scale_nr = random.sample([0.50, 0.75, 1, 1.25, 1.5], k=1)[0] # Random scale by which to change image size
    dim = x.shape[2]
    out_dim = int(dim*scale_nr)
    x = torch.unsqueeze(x, 0)
    y = torch.unsqueeze(y,0).float()
    x = torch.nn.functional.interpolate(x, (out_dim,out_dim,out_dim), mode='trilinear')
    y = torch.nn.functional.interpolate(y, (out_dim,out_dim,out_dim), mode='trilinear')
    x = torch.squeeze(x, 0)
    y = torch.squeeze(y, 0)
    y = torch.squeeze(y, 0).int()
plt.subplots_adjust(wspace=0, hspace=0)
plot(x,y,2,6,6,6,'Scaling')
plt.show()

plt.figure()
plot(brain_region1,brain_region[0,:],1,2,1,1,'Brain Region')
plt.show()
