import torch
import numpy as np
import random
import elasticdeform

###################################################################################
# Data augmentation code to be run during training. Input in the form x=[B,C,H,W,D], y = [B,H,W,D] where x is the batch of preprocessed
# mri scans and y is the batch of labels.
###################################################################################

class DataAugment():
    def __init__(self, x, y):
        self.scans = x
        self.mask = y
        self.batch_size = x.shape[0]

    def elastic_deform(self,X,Y):
        X = X.numpy()
        Y = Y.numpy()
        brain_region = (X > 0).astype('float') * 10 # Multiplying by 10 so there would be a bigger difference between foreground and background pixels to avoid voxels
        # being assigned the wrong label after the elastic deformation

        #Split the labels and deform them separately, as if done together together they'll get mixed up. 10 times multiplication again.
        lbl1 = (Y == 1) * 10.0
        lbl2 = (Y == 2) * 10.0
        lbl3 = (Y == 3) * 10.0
        sigma_nr = random.uniform(0, 7) # Random factor by which to deform - not too high
        deform_axes = tuple(sorted(random.sample([1,2,3], k = 2))) # Random two axes in which to deform. Two to save on computational time. Need to sort the array as axis
        #variable requires sorted input

        [X, brain_region, lbl1, lbl2, lbl3] = elasticdeform.deform_random_grid([X, brain_region, lbl1, lbl2, lbl3], axis=deform_axes, sigma=sigma_nr, points=3)

        brain_region = brain_region.astype('int') > 0
        X = X * brain_region # To make sure background pixels remain 0 in the scans
        X[X<0] = 0 # Remove any negative values - background values close 0

        lbl1[lbl1 < 3] = -1
        lbl2[lbl2 < 3] = -1
        lbl3[lbl3 < 3] = -1
        Y = np.argmax([np.zeros((1, 128, 128, 128)), lbl1, lbl2, lbl3], axis=0).astype('long')
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)

        return X,Y

    def rotate(self,X,Y):
        rotate_nr = random.randint(0, 3)  # Number of times to rotate
        rotate_dir = random.sample([1, 2, 3], k=2)  # Direction of axis in which to rotate
        X = torch.rot90(X, rotate_nr, rotate_dir)
        Y = torch.rot90(Y, rotate_nr, rotate_dir)

        return X, Y

    def flip(self,X,Y):
        flip_dir = random.sample([1, 2, 3], k=2)  # Random axis direction in which to flip
        X = torch.flip(X, flip_dir)
        Y = torch.flip(Y, flip_dir)

        return X,Y

    def gamma_correction(self,X):
        gamma = random.uniform(0.5, 1.5)  # Random gamma value by which to correct
        X = torch.pow(X, gamma)

        return X

    def scale(self,X,Y):
        scale_nr = random.sample([0.50, 0.75, 1, 1.25, 1.5], k=1)[0]  # Random scale by which to change image size
        in_dim = X.shape[2]
        out_dim = int(in_dim * scale_nr) # New size to scale to
        # Scale labels separately again
        Y = torch.unsqueeze(Y, 0)
        lbl1 = (Y == 1) * 10.0
        lbl2 = (Y == 2) * 10.0
        lbl3 = (Y == 3) * 10.0
        X = torch.nn.functional.interpolate(X, (out_dim, out_dim, out_dim), mode='trilinear')
        lbl1 = torch.nn.functional.interpolate(lbl1, (out_dim, out_dim, out_dim), mode='trilinear')
        lbl2 = torch.nn.functional.interpolate(lbl2, (out_dim, out_dim, out_dim), mode='trilinear')
        lbl3 = torch.nn.functional.interpolate(lbl3, (out_dim, out_dim, out_dim), mode='trilinear')
        lbl1 = lbl1.squeeze(0)
        lbl2 = lbl2.squeeze(0)
        lbl3 = lbl3.squeeze(0)
        lbl1[lbl1 < 3] = -1
        lbl2[lbl2 < 3] = -1
        lbl3[lbl3 < 3] = -1
        _, Y = torch.stack([torch.zeros(self.batch_size,out_dim, out_dim, out_dim), lbl1, lbl2, lbl3],dim=0).max(0)
        Y = Y.long()
        return X, Y

    def augment(self):

        # Except for scaling, perform the data augmentations separately for every person in the batch
        for person in range(self.batch_size):
            x = self.scans[person]
            y = self.mask[person]
            y = torch.unsqueeze(y,0)  # As x is in 4D, make y 4D too so the scans and the labels can be inputted together to the various functions


            # Random elastic deformation
            if(random.random() > 0.5):
                x, y = self.elastic_deform(x, y)

            # Random rotation
            if(random.random() > 0.5):
                x,y = self.rotate(x,y)

            # Random flip
            if(random.random() > 0.5):
                x,y = self.flip(x,y)

            # Random gamma correction
            if(random.random() > 0.5):
                x = self.gamma_correction(x)

            y = torch.squeeze(y, 0)

            self.scans[person] = x
            self.mask[person] = y

        # Random scaling
        if (random.random() > 0.5):
            self.scans, self.mask = self.scale(self.scans, self.mask)

        return self.scans, self.mask
