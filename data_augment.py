import torch
import numpy as np
import random
import elasticdeform

###################################################################################
# Data augmentation code to be run during training.
# INPUT:
#   x = [B,C,H,W,D] torch tensor containing B batches of C different modalities.
#   y = [B,H,W,D] torch tensor containing B batches of segmentation labels
#   augmentations_to_use: list containing the following possible values: 'Elastic', 'Rotate', 'Flip', 'Gamma', 'Noise', 'Scale'.
#                         If nothing is given then all are used.
#
# OUTPUT:
#   Torch tensors x and y augmented with the specified data augmentations, each augmentation applied with 0.5 probability.
###################################################################################

class DataAugment(object):
    def __init__(self, x, y, augmentations_to_use=None):
        if augmentations_to_use is None:
            augmentations_to_use = ['Elastic', 'Rotate', 'Flip', 'Gamma', 'Noise', 'Scale']
        self.scans = x
        self.mask = y
        self.batch_size = x.shape[0]
        self.augmentations = augmentations_to_use

    def elastic_deform(self, X, Y):
        # Elastic deformation with a random square deformation grid, where the displacements are sampled from a normal distribution with
        # standard deviation sigma. Applies elastic deformation in all 3 axes, if you wish to speed up training time change the d variable
        # to two axes chosen randomly. Uses elasticdeform package from gvtulder/elasticdeform.

        X = X.numpy()
        Y = Y.numpy()
        brain_region = (X > 0).astype('float') * 10 # Multiplying by 10 so there would be a bigger difference between foreground and background pixels to avoid voxels
        # being assigned the wrong label after the elastic deformation

        #Split the labels and deform them separately, as if done together together they'll get mixed up. 10 times multiplication again.
        lbl1 = (Y == 1) * 10.0
        lbl2 = (Y == 2) * 10.0
        lbl3 = (Y == 3) * 10.0
        sigma_nr = 5 # Random factor by which to deform
        d = (1,2,3) # Axes in which to deform
        # d = tuple(sorted(random.sample([1, 2, 3], k=2))) # Use this instead if you wish to speed up training.

        [X, brain_region, lbl1, lbl2, lbl3] = elasticdeform.deform_random_grid([X, brain_region, lbl1, lbl2, lbl3], axis=[d]*5, sigma=sigma_nr, points=3)

        brain_region = brain_region.astype('int') > 0
        X = X * brain_region  # To make sure background pixels remain 0 in the scans
        X[X < 0] = 0  # Remove any negative values - background values close 0

        lbl1[lbl1 < 3] = -1
        lbl2[lbl2 < 3] = -1
        lbl3[lbl3 < 3] = -1
        Y = np.argmax([np.zeros((1, 128, 128, 128)), lbl1, lbl2, lbl3], axis=0).astype('long')
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)

        return X, Y

    def rotate(self, X, Y):
        # Rotates by 90 degrees randomly chosen k=[1,2,3] times, from one axis towards another. Axes are randomly chosen.
        rotate_nr = random.randint(1, 3)  # Number of times to rotate
        rotate_dir = random.sample([1, 2, 3], k=2)  # Direction of axis in which to rotate
        X = torch.rot90(X, rotate_nr, rotate_dir)
        Y = torch.rot90(Y, rotate_nr, rotate_dir)

        return X, Y

    def flip(self, X, Y):
        #Flip horizontally (mirroring). x dimension is axis 1.
        X = torch.flip(X, [1])
        Y = torch.flip(Y, [1])
        return X, Y

    def gamma_correction(self, X):
        # Gamma correction according to Vout = A*Vin^(gamma), where A=1. Gamma uniformly sampled from range [0.5].
        gamma = random.uniform(0.5, 1.5)  # Random gamma value by which to correct
        X = torch.pow(X, gamma)
        return X

    def random_noise(self, X):
        #Add random noise to the model, where noise is square root of 0.1 times a uniform distribution.
        X = X + (0.1**0.5)*torch.randn(X.shape)
        return X

    def scale(self,X,Y):
        # Scale by in all axes by one of randomly chosen scaling factors [0.5,0.75,1.25,1.5]
        scale_nr = random.sample([0.50, 0.75, 1, 1.25, 1.5], k=1)[0]  # Random scale by which to change image size
        in_dim = X.shape[2]
        out_dim = int(in_dim * scale_nr) # New size to scale to
        Y = torch.unsqueeze(Y,0)
        # Scale labels separately again
        lbl1 = (Y == 1) * 10.0
        lbl2 = (Y == 2) * 10.0
        lbl3 = (Y == 3) * 10.0
        X = torch.nn.functional.interpolate(X, (out_dim, out_dim, out_dim), mode='trilinear')
        lbl1 = torch.nn.functional.interpolate(lbl1, (out_dim, out_dim, out_dim), mode='trilinear')
        lbl2 = torch.nn.functional.interpolate(lbl2, (out_dim, out_dim, out_dim), mode='trilinear')
        lbl3 = torch.nn.functional.interpolate(lbl3, (out_dim, out_dim, out_dim), mode='trilinear')
        lbl1[lbl1 < 3] = -1
        lbl2[lbl2 < 3] = -1
        lbl3[lbl3 < 3] = -1
        _, Y = torch.stack([torch.zeros(1,self.batch_size, out_dim, out_dim, out_dim), lbl1, lbl2, lbl3], dim=0).max(0)
        Y = torch.squeeze(Y,0)
        Y = Y.long()
        return X, Y

    def augment(self):
        # Except for scaling, perform the data augmentations separately for every person in the batch
        to_scale = False
        for person in range(self.batch_size):
            x = self.scans[person]
            y = self.mask[person]
            y = torch.unsqueeze(y, 0)  # As x is in 4D, make y 4D too so the scans and the labels can be inputted together to the various functions

            for augmentation in self.augmentations:
                if(augmentation == 'Elastic'):
                    if(random.random() > 0.5):
                        x, y = self.elastic_deform(x, y)
                elif(augmentation == 'Rotate'):
                    if(random.random() > 0.5):
                        x, y = self.rotate(x,y)
                elif(augmentation == 'Flip'):
                    if(random.random() > 0.5):
                        x, y = self.flip(x,y)
                elif(augmentation == 'Gamma'):
                    if(random.random() > 0.5):
                        x = self.gamma_correction(x)
                elif(augmentation == 'Noise'):
                    if (random.random() > 0.5):
                        x = self.random_noise(x)
                elif(augmentation == 'Scale'):
                    to_scale = True

            y = torch.squeeze(y, 0)
            self.scans[person] = x
            self.mask[person] = y

        if to_scale:
            if (random.random() > 0.5):
                self.scans, self.mask = self.scale(self.scans, self.mask)

        return self.scans, self.mask
