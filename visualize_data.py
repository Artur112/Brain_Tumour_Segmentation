import nibabel as nib
from matplotlib import pyplot as plt
from skimage.transform import resize
# from batchviewer import view_batch
import numpy as np
import torch.nn.functional as F
import os
import shutil
import ants
import time
from data_loaders import Dataset
from torch.utils import data
from data_augment import DataAugment
import random
import torch
import nibabel as nib
import elasticdeform
import sys
"""
address = 'C:/users/Artur/Desktop/UCL/Brats2019/Data/MIDL/Training_Data/Augmented'

for subdir in os.listdir(address):
    fileid = os.listdir(os.path.join(address,subdir))
    for fd in fileid:
        name = fd[:-7]
        nfbs_name = name[-9:]
        brats_name = name[:19]
        scan_name = name[24:]
        scan_name = scan_name[:-15]
        os.rename(os.path.join(address,subdir,fd), os.path.join(address,subdir, "{}_reg_{}_{}.nii.gz".format(brats_name,nfbs_name,scan_name)))
    os.rename(os.path.join(address,subdir), os.path.join(address,"{}_reg_{}".format(brats_name,nfbs_name)))
    #shutil.copy(os.path.join(address, subdir)+ "/{}_t1.nii.gz".format(subdir), 'C:/users/artur/Desktop/UCL/Brats2019/Data/MIDL/Training_Data/fslmerge-testing')
"""
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


####################### First make the transformation matrix of the SRI-24 template match that of the Brats Data ####################
# brats_img = nib.load('/home/artur-cmic/Desktop/UCL/Brats2019/Data/MICCAI_BraTS_2019_Data_Training/data/HGG/BraTS19_CBICA_ABB_1/BraTS19_CBICA_ABB_1_t1.nii.gz')
#brats_header = brats_img.get_header()
#brats_affine = brats_img.get_affine()
#print(nib.aff2axcodes(brats_img.affine))

#template = nib.load('/home/artur-cmic/Desktop/UCL/Brats2019/Data/sri24_spm8/templates/template.nii.gz').get_fdata()
#print(nib.aff2axcodes(template.affine))
#template.header.set_sform(np.diag([-1,-1,1,1]))
#9print(nib.aff2axcodes(template.affine))
#Tmatrix = [[-1,-0,-0,0],[-0,-1,-0,239],[0,0,1,0],[0,0,0,1]]
#img = nib.Nifti1Image(template, brats_affine)
#nib.save(img, 'SRI-24-template.nii.gz')

######################## Registering all of the NFBS dataset images to the SRI-24 template with affine only transforms ######################

#template = nib.load('SRI-24-template.nii.gz')
#template = ants.from_nibabel(template)

# Get folder paths and ids of all of the NFBS images
#folder_paths = []
#folder_ids = []
#for subdir in os.listdir('/home/artur-cmic/Desktop/UCL/Brats2019/Data/NFBS_Dataset'):
#    folder_paths.append(os.path.join('/home/artur-cmic/Desktop/UCL/Brats2019/Data/NFBS_Dataset', subdir))
#    folder_ids.append(subdir)
#start_time = time.time()

#for idx in range(1): #len(folder_ids)):
#    moving = nib.load('{}/sub-{}_ses-NFB3_T1w_brain.nii.gz'.format(folder_paths[idx], folder_ids[idx]))
#    moving = ants.from_nibabel(moving)
#    mytx = ants.registration(fixed=template, moving=moving, type_of_transform='Affine')
#    warped_moving = mytx['warpedmovout']
#    mywarpedimage = ants.apply_transforms(fixed=template, moving=moving,transformlist=mytx['fwdtransforms'])
#    nib.save(mywarpedimage.to_nibabel(), '/home/artur-cmic/Desktop/UCL/Brats2019/Data/NFBS_to_template/{}.nii.gz'.format(folder_ids[idx]))
#    print('Registered {}/{} NFBS image to template'.format(idx+1, len(folder_ids)))
#elapsed_time = time.time() - start_time
#print("Registering linearly to template took {}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
################################################################################################################################################
"""
start_time = time.time()
t1_img = nib.load("BraTS19_CBICA_APZ_1_t1.nii.gz")
t1_np = nib.load("BraTS19_CBICA_APZ_1_t1.nii.gz").get_fdata()
seg_img = nib.load("BraTS19_CBICA_APZ_1_seg.nii.gz").get_data().astype('long')
brats_affine = t1_img.get_affine()
t1_img = ants.from_nibabel(t1_img)
seg_img[t1_np>0] = -1
mask = (seg_img == 0).astype('uint8')
print(seg_img.shape)
mask = ants.from_nibabel(nib.Nifti1Image(mask, brats_affine))
nfbs_img = ants.from_nibabel(nib.load("NFBS2BRATS.nii.gz"))
mytx_withmask = ants.registration(fixed=nfbs_img, moving=t1_img, mask=mask, type_of_transform='SyNCC')
mytx_nomask = ants.registration(fixed=nfbs_img, moving=t1_img, type_of_transform='SyNCC')
warped_t1_withmask = ants.apply_transforms(fixed=nfbs_img, moving=t1_img, transformlist=mytx_withmask['fwdtransforms']).to_nibabel()
warped_t1_nomask = ants.apply_transforms(fixed=nfbs_img, moving=t1_img, transformlist=mytx_nomask['fwdtransforms']).to_nibabel()
nib.save(warped_t1_withmask, "brats_2_nfbs_SynCC_withmask.nii.gz")
nib.save(warped_t1_nomask, "brats_2_nfbs_SynCC_nomask.nii.gz")


labels = nib.load("BraTS19_CBICA_APZ_1_seg.nii.gz").get_data().astype('long')
labels[labels == 4] = 3
lbl1 = ants.from_nibabel(nib.Nifti1Image(((labels == 1) * 10).astype('uint8'), brats_affine))
lbl2 = ants.from_nibabel(nib.Nifti1Image(((labels == 2) * 10).astype('uint8'), brats_affine))
lbl3 = ants.from_nibabel(nib.Nifti1Image(((labels == 3) * 10).astype('uint8'), brats_affine))
warped_lbl1_withmask = ants.apply_transforms(fixed=nfbs_img, moving=lbl1, transformlist=mytx_withmask['fwdtransforms']).to_nibabel().get_fdata()
warped_lbl2_withmask = ants.apply_transforms(fixed=nfbs_img, moving=lbl2, transformlist=mytx_withmask['fwdtransforms']).to_nibabel().get_fdata()
warped_lbl3_withmask = ants.apply_transforms(fixed=nfbs_img, moving=lbl3, transformlist=mytx_withmask['fwdtransforms']).to_nibabel().get_fdata()

warped_lbl1_nomask = ants.apply_transforms(fixed=nfbs_img, moving=lbl1, transformlist=mytx_nomask['fwdtransforms']).to_nibabel().get_fdata()
warped_lbl2_nomask = ants.apply_transforms(fixed=nfbs_img, moving=lbl2, transformlist=mytx_nomask['fwdtransforms']).to_nibabel().get_fdata()
warped_lbl3_nomask = ants.apply_transforms(fixed=nfbs_img, moving=lbl3, transformlist=mytx_nomask['fwdtransforms']).to_nibabel().get_fdata()
# Remove uncertain pixels at the edges of a label area before merging labels. Essentially remove pixels with class belonging confidence of < 30%.
# Made equal to -1 instead of 0, just to make sure that the background pixels are always assigned label 0 with argmax.
warped_lbl1_withmask[warped_lbl1_withmask < 3] = -1
warped_lbl2_withmask[warped_lbl2_withmask < 3] = -1
warped_lbl3_withmask[warped_lbl3_withmask < 3] = -1
warped_lbl1_nomask[warped_lbl1_nomask < 3] = -1
warped_lbl2_nomask[warped_lbl2_nomask < 3] = -1
warped_lbl3_nomask[warped_lbl3_nomask < 3] = -1
# Merge labels by taking argmax of label values - for a pixel that belongs to two classes after resize, assign to the class
# that its value is highest for. Adding np.zeros to the first dimension so np.argmax would give 1 for label 1 and not 0.
warped_labels_withmask = np.argmax(np.asarray([np.zeros((240, 240, 155)), warped_lbl1_withmask, warped_lbl2_withmask, warped_lbl3_withmask]), axis=0).astype('uint8')
warped_labels_withmask = nib.Nifti1Image(warped_labels_withmask, brats_affine)

warped_labels_nomask = np.argmax(np.asarray([np.zeros((240, 240, 155)), warped_lbl1_nomask, warped_lbl2_nomask, warped_lbl3_nomask]), axis=0).astype('uint8')
warped_labels_nomask = nib.Nifti1Image(warped_labels_nomask, brats_affine)

nib.save(warped_labels_withmask, "brats_2_nfbs_SynCC_seg_withmask.nii.gz")
nib.save(warped_labels_nomask, "brats_2_nfbs_SynCC_seg_nomask.nii.gz")

elapsed_time = time.time() - start_time
print("Registering took {}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
"""
### Understanding how torch.flip works #####################
#orig = nib.load('BraTS19_CBICA_APZ_1_t1.nii.gz')
#affine = orig.get_affine()
#orig = torch.from_numpy(orig.get_fdata())
#print(orig.shape)
#flip_dir = [0]
#flipped = torch.flip(orig, flip_dir)
#nib.save(nib.Nifti1Image(flipped.numpy(),affine), "{}.nii.gz".format(flip_dir))

"""
def normalize_inputs(X):
    if isinstance(X, np.ndarray):
        Xs = [X]
    elif isinstance(X, list):
        Xs = X
    else:
        raise Exception('X should be a numpy.ndarray or a list of numpy.ndarrays.')

    # check X inputs
    assert len(Xs) > 0, 'You must provide at least one image.'
    assert all(isinstance(x, np.ndarray) for x in Xs), 'All elements of X should be numpy.ndarrays.'
    return Xs

def normalize_axis_list(axis, Xs):
    if axis is None:
        axis = [tuple(range(x.ndim)) for x in Xs]
    elif isinstance(axis, int):
        axis = (axis,)
    if isinstance(axis, tuple):
        axis = [axis] * len(Xs)
    assert len(axis) == len(Xs), 'Number of axis tuples should match number of inputs.'
    input_shapes = []
    for x, ax in zip(Xs, axis):
        assert isinstance(ax, tuple), 'axis should be given as a tuple'
        assert all(isinstance(a, int) for a in ax), 'axis must contain ints'
        assert len(ax) == len(axis[0]), 'All axis tuples should have the same length.'
        assert ax == tuple(set(ax)), 'axis must be sorted and unique'
        assert all(0 <= a < x.ndim for a in ax), 'invalid axis for input'
        input_shapes.append(tuple(x.shape[d] for d in ax))
    assert len(set(input_shapes)) == 1, 'All inputs should have the same shape.'
    deform_shape = input_shapes[0]
    return axis, deform_shape

def compute_output_shapes(Xs, axis, deform_shape, crop):
    if crop is not None:
        assert isinstance(crop, (tuple, list)), "crop must be a tuple or a list."
        assert len(crop) == len(deform_shape)
        output_shapes = [list(x.shape) for x in Xs]
        output_offset = [0 for d in range(len(axis[0]))]
        for d in range(len(axis[0])):
            if isinstance(crop[d], slice):
                assert crop[d].step is None
                start = (crop[d].start or 0)
                stop = (crop[d].stop or deform_shape[d])
                assert start >= 0
                assert start < stop and stop <= deform_shape[d]
                for i in range(len(Xs)):
                    output_shapes[i][axis[i][d]] = stop - start
                if start > 0:
                    output_offset[d] = start
            else:
                raise Exception('Crop must be a slice.')
        if any(o > 0 for o in output_offset):
            output_offset = np.array(output_offset).astype('int64')
        else:
            output_offset = None
    else:
        output_shapes = [x.shape for x in Xs]
        output_offset = None
    return output_shapes, output_offset

def normalize_displacement(displacement, Xs, axis):
    assert isinstance(displacement, np.ndarray), 'Displacement matrix should be a numpy.ndarray.'
    assert displacement.ndim == len(axis[0]) + 1, 'Number of dimensions of displacement does not match input.'
    assert displacement.shape[0] == len(axis[0]), 'First dimension of displacement should match number of input dimensions.'
    return displacement

def normalize_order(order, Xs):
    if not isinstance(order, (tuple, list)):
        order = [order] * len(Xs)
    assert len(Xs) == len(order), 'Number of order parameters should be equal to number of inputs.'
    assert all(0 <= o and o <= 5 for o in order), 'order should be 0, 1, 2, 3, 4 or 5.'
    return np.array(order).astype('int64')

def normalize_mode(mode, Xs):
    if not isinstance(mode, (tuple, list)):
        mode = [mode] * len(Xs)
    mode = [extend_mode_to_code(o) for o in mode]
    assert len(Xs) == len(mode), 'Number of mode parameters should be equal to number of inputs.'
    return np.array(mode).astype('int64')

def normalize_cval(cval, Xs):
    if not isinstance(cval, (tuple, list)):
        cval = [cval] * len(Xs)
    assert len(Xs) == len(cval), 'Number of cval parameters should be equal to number of inputs.'
    return np.array(cval).astype('float64')

def extend_mode_to_code(mode):
    #Convert an extension mode to the corresponding integer code.
    
    if mode == 'nearest':
        return 0
    elif mode == 'wrap':
        return 1
    elif mode == 'reflect':
        return 2
    elif mode == 'mirror':
        return 3
    elif mode == 'constant':
        return 4
    else:
        raise RuntimeError('boundary mode not supported')

########## Creating Elastically Deformed Scans ########################3

preprocessed_data_path = r'/home/artur-cmic/Desktop/UCL/MIDL/Training_Data/Originals_Preprocessed'
save_data_path = r'/home/artur-cmic/Desktop/UCL/MIDL/Training_Data/Elastically_Deformed_Preprocessed'
save_raw_path = r'/home/artur-cmic/Desktop/UCL/MIDL/Training_Data/Elastically_Deformed'

# Get paths and names (IDS) of folders that store the multimodal training data
folder_paths = []
folder_ids = []
for subdir in os.listdir(preprocessed_data_path):
    folder_paths.append(os.path.join(preprocessed_data_path, subdir))
    folder_ids.append(subdir)

augmentations_to_use = ['Elastic']
orig_affine = nib.load("/home/artur-cmic/Desktop/UCL/MIDL/Training_Data/Originals/BraTS19_CBICA_AWV_1/BraTS19_CBICA_AWV_1_t1.nii.gz").get_affine()
ii = 1
for patient in range(len(folder_paths)):
    if os.path.isdir(os.path.join(save_data_path, folder_ids[patient])):
        shutil.rmtree("{}/{}".format(save_data_path, folder_ids[patient]))
        shutil.rmtree("{}/{}".format(save_raw_path, folder_ids[patient]))

    os.mkdir(os.path.join(save_data_path, folder_ids[patient]))
    if patient == 0:
        os.mkdir(os.path.join(save_raw_path, folder_ids[patient]))
    scans = np.load(r"{}/{}_scans.npy".format(folder_paths[patient], folder_ids[patient]))
    masks = np.expand_dims(np.load(r"{}/{}_mask.npy".format(folder_paths[patient], folder_ids[patient])), 0)
    a = len(os.listdir(os.path.join(save_data_path,folder_ids[patient])))
    while a < 31:
        ########################## Elastic Deformation to create anatomically incorrect examples ###############################################
        # Data Augment
        brain_region = (scans > 0).astype('float') * 10  # Multiplying by 10 so there would be a bigger difference between foreground and background pixels to avoid voxels
        # being assigned the wrong label after the elastic deformation

        # Split the labels and deform them separately, as if done together together they'll get mixed up. 10 times multiplication again.
        lbl1 = (masks == 1) * 10.0
        lbl2 = (masks == 2) * 10.0
        lbl3 = (masks == 3) * 10.0
        X = [scans, brain_region, lbl1, lbl2, lbl3]
        mu = 0 #30
        sigma = 8
        # d = tuple(sorted(random.sample([1,2,3], k = 2))) # Random two axes in which to deform. Two to save on computational time. Need to sort the array as axis
        points = 3
        order = 3
        mode = 'constant'
        cval = 0.0
        crop = None
        prefilter = True
        axes = (1, 2, 3)
        Xs = normalize_inputs(X)
        axis, deform_shape = normalize_axis_list(axes, Xs)
        if not isinstance(points, (list, tuple)):
            points = [points] * len(deform_shape)

        displacement = np.random.randn(len(deform_shape), *points) * sigma + mu
        [X, brain_region, lbl1, lbl2, lbl3] = elasticdeform.deform_grid(X, displacement, order, mode, cval, crop, prefilter, axis)

        brain_region = brain_region.astype('int') > 0
        X = X * brain_region  # To make sure background pixels remain 0 in the scans
        X[X < 0] = 0  # Remove any negative values - background values close 0

        lbl1[lbl1 < 3] = -1
        lbl2[lbl2 < 3] = -1
        lbl3[lbl3 < 3] = -1
        Y = np.argmax([np.zeros((1, 128, 128, 128)), lbl1, lbl2, lbl3], axis=0).astype('long')
        ####################################################################################################################

        if not os.path.isdir("{}/{}/{}_{}".format(save_data_path, folder_ids[patient], folder_ids[patient], a)):
            batchh = X
            labelss = np.squeeze(Y,0)
            os.mkdir("{}/{}/{}_{}".format(save_data_path, folder_ids[patient], folder_ids[patient], a))
            np.save("{}/{}/{}_{}/{}_{}_scans.npy".format(save_data_path, folder_ids[patient], folder_ids[patient], a, folder_ids[patient], a), batchh)
            np.save("{}/{}/{}_{}/{}_{}_mask.npy".format(save_data_path, folder_ids[patient], folder_ids[patient], a,folder_ids[patient], a), labelss)

            if patient == 0:
                os.mkdir("{}/{}/{}_{}".format(save_raw_path, folder_ids[patient], folder_ids[patient], a))

                t1 = batchh[0]
                t1ce = batchh[1]
                t2 = batchh[2]
                flair = batchh[3]

                t1 = resize(t1, (240, 240, 155), preserve_range=True, anti_aliasing=True)
                t1ce = resize(t1ce, (240, 240, 155), preserve_range=True, anti_aliasing=True)
                t2 = resize(t2, (240, 240, 155), preserve_range=True, anti_aliasing=True)
                flair = resize(flair, (240, 240, 155), preserve_range=True, anti_aliasing=True)

                lbl1 = resize((labelss == 1) * 10, (240, 240, 155), preserve_range=True, anti_aliasing=True)
                lbl2 = resize((labelss == 2) * 10, (240, 240, 155), preserve_range=True, anti_aliasing=True)
                lbl3 = resize((labelss == 3) * 10, (240, 240, 155), preserve_range=True, anti_aliasing=True)

                # Remove uncertain pixels at the edges of a label area before merging labels. Essentially remove pixels with class belonging confidence of < 30%.
                # Made equal to -1 instead of 0, just to make sure that the background pixels are always assigned label 0 with argmax.
                lbl1[lbl1 < 3] = -1
                lbl2[lbl2 < 3] = -1
                lbl3[lbl3 < 3] = -1

                # Merge labels by taking argmax of label values - for a pixel that belongs to two classes after resize, assign to the class
                # that its value is highest for. Adding np.zeros to the first dimension so np.argmax would give 1 for label 1 and not 0.
                img_segm = np.argmax(np.asarray([np.zeros((240, 240, 155)), lbl1, lbl2, lbl3]), axis=0).astype(np.uint8)

                img_segm[img_segm == 3] = 4

                nib.save(nib.Nifti1Image(t1, orig_affine),"{}/{}/{}_{}/{}_{}_t1.nii.gz".format(save_raw_path, folder_ids[patient], folder_ids[patient], a,  folder_ids[patient], a))
                nib.save(nib.Nifti1Image(t1ce, orig_affine),"{}/{}/{}_{}/{}_{}_t1ce.nii.gz".format(save_raw_path, folder_ids[patient], folder_ids[patient], a,  folder_ids[patient], a))
                nib.save(nib.Nifti1Image(t2, orig_affine), "{}/{}/{}_{}/{}_{}_t2.nii.gz".format(save_raw_path, folder_ids[patient], folder_ids[patient], a,  folder_ids[patient], a))
                nib.save(nib.Nifti1Image(flair, orig_affine),"{}/{}/{}_{}/{}_{}_flair.nii.gz".format(save_raw_path, folder_ids[patient], folder_ids[patient], a,  folder_ids[patient], a))
                nib.save(nib.Nifti1Image(img_segm, orig_affine),"{}/{}/{}_{}/{}_{}_seg.nii.gz".format(save_raw_path, folder_ids[patient], folder_ids[patient], a,  folder_ids[patient], a))
        a = len(os.listdir(os.path.join(save_data_path, folder_ids[patient])))
    print("Deformed {}/{} scans".format(ii, len(folder_paths)))
    ii = ii + 1

#orig = nib.load("/home/artur-cmic/Desktop/UCL/MIDL/Training_Data/Originals/BraTS19_CBICA_AWV_1/BraTS19_CBICA_AWV_1_t1.nii.gz").get_affine()

"""
"""
##############################################
raw_data_path = r"/home/artur-cmic/Desktop/UCL/MIDL/Data/James_Data/Originals"
save_preprocessed_data_path = r"/home/artur-cmic/Desktop/UCL/MIDL/Data/James_Data/Originals_Compatible"


# Create folder to store preprocessed data in, exit if folder already exists.
if not os.path.isdir(save_preprocessed_data_path):
    os.mkdir(save_preprocessed_data_path)

# Get folder paths and ids of where the raw scans are stored
folder_paths = []
folder_IDS = []
for subdir in os.listdir(raw_data_path):
    folder_paths.append(os.path.join(raw_data_path, subdir))
    folder_IDS.append(subdir)
orig_affine = nib.load("/home/artur-cmic/Desktop/UCL/MIDL/Data/Training_Data/Originals/BraTS19_CBICA_AWV_1/BraTS19_CBICA_AWV_1_t1.nii.gz").get_affine()
i = 1
for patient in range(len(folder_paths)):
    data_folder = folder_paths[patient]
    data_id = folder_IDS[patient]
    os.mkdir(os.path.join(save_preprocessed_data_path, data_id))

    output_size = (240,240,155) # Size to change images to
    # Load in and resize the mri images
    img_t1 = resize(nib.load(os.path.join(data_folder, data_id) + "_0001.nii.gz").get_fdata(),output_size)
    img_t1ce = resize(nib.load(os.path.join(data_folder, data_id) + "_0002.nii.gz").get_fdata(),output_size)
    img_t2 = resize(nib.load(os.path.join(data_folder, data_id) + "_0003.nii.gz").get_fdata(),output_size)
    img_flair = resize(nib.load(os.path.join(data_folder, data_id) + "_0000.nii.gz").get_fdata(),output_size)

    img_segm = nib.load(os.path.join(data_folder, data_id) + "_Segmentation.nii.gz").get_fdata().astype('long')
    img_segm = img_segm[:,:,:,0]
    # Segmentation Mask has labels 0,1,2,4. Will change these to 0,1,2,3 and perform resizing on the labels separately
    # Combine them afterwards. Multiplication of labels by 10 so difference between label and background pixel would be
    # greater, otherwise resize wont work properly.
    #img_segm[img_segm == 4] = 3
    nonenhancing = resize((img_segm == 3)*10, output_size, preserve_range=True, anti_aliasing=True)
    edema = resize((img_segm == 1)*10, output_size, preserve_range=True, anti_aliasing=True)
    enhancing = resize((img_segm == 2)*10, output_size, preserve_range=True, anti_aliasing=True)

    # Remove uncertain pixels at the edges of a label area before merging labels. Essentially remove pixels with class belonging confidence of < 30%.
    # Made equal to -1 instead of 0, just to make sure that the background pixels are always assigned label 0 with argmax.
    nonenhancing[nonenhancing < 3] = -1
    edema[edema < 3] = -1
    enhancing[enhancing< 3] = -1

    # Merge labels by taking argmax of label values - for a pixel that belongs to two classes after resize, assign to the class
    # that its value is highest for. Adding np.zeros to the first dimension so np.argmax would give 1 for label 1 and not 0.
    img_segm = np.argmax(np.asarray([np.zeros(output_size),nonenhancing, edema, enhancing]),axis=0).astype('uint8')
    img_segm[img_segm == 3] = 4
    nib.save(nib.Nifti1Image(img_t1, orig_affine), "{}/{}/{}_t1.nii.gz".format(save_preprocessed_data_path, data_id, data_id))
    nib.save(nib.Nifti1Image(img_t1ce, orig_affine), "{}/{}/{}_t1ce.nii.gz".format(save_preprocessed_data_path, data_id, data_id))
    nib.save(nib.Nifti1Image(img_t2, orig_affine),"{}/{}/{}_t2.nii.gz".format(save_preprocessed_data_path, data_id, data_id))
    nib.save(nib.Nifti1Image(img_flair, orig_affine),"{}/{}/{}_flair.nii.gz".format(save_preprocessed_data_path, data_id, data_id))
    nib.save(nib.Nifti1Image(img_segm, orig_affine), "{}/{}/{}_seg.nii.gz".format(save_preprocessed_data_path, data_id, data_id))
    print("Preprocessed patient {}/{} scans".format(i, len(folder_paths)))
    i = i + 1
"""

