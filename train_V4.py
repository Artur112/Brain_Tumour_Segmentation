import torch
import numpy as np
import random
import os
from torch.utils import data
from sklearn.model_selection import KFold
from data_loaders import Dataset
from data_augment import DataAugment
from models import Modified3DUNet
import torch.nn as nn
import time
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import io
import PIL.Image
from torchvision.transforms import ToTensor


def dice(out, target):
    num = (out*target).sum()
    den = out.sum() + target.sum()
    return - (2/n_classes)*num/den

def gen_plot(scans, mask, prediction):
    _, indices = prediction.max(0)
    indices = indices.cpu().detach().numpy()
    prediction = prediction.cpu().detach().numpy()
    slices = [30,60,90] # Slices to display
    plt.figure()
    for row in range(1,4):
        plt.subplot(3, 7, 1 + (row-1)*7)
        #Showing the t1ce scan
        plt.imshow(scans[1, :, :, slices[row-1]], cmap='gray')
        plt.ylabel("Slice {}".format(slices[row-1]))
        if row==1:
            plt.title('T1ce Scan')
        plt.subplot(3, 7, 2 + (row-1)*7)
        plt.imshow(mask[:, :, slices[row-1]])
        if row==1:
            plt.title('Mask')
        plt.subplot(3, 7, 3 + (row-1)*7)
        plt.imshow(prediction[0,:,:,slices[row-1]])
        if row==1:
            plt.title('Cl 0')

        plt.subplot(3, 7, 4 + (row - 1) * 7)
        plt.imshow(prediction[1, :, :, slices[row - 1]])
        if row == 1:
            plt.title('Cl 1')

        plt.subplot(3, 7, 5 + (row - 1) * 7)
        plt.imshow(prediction[2, :, :, slices[row - 1]])
        if row == 1:
            plt.title('Cl 2')

        plt.subplot(3, 7, 6 + (row - 1) * 7)
        plt.imshow(prediction[3, :, :, slices[row - 1]])
        if row == 1:
            plt.title('Cl 3')

        plt.subplot(3, 7, 7 + (row - 1) * 7)
        plt.imshow(indices[:, :, slices[row-1]])
        if row == 1:
            plt.title('Pred')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    return image


##############################################
# Code for training the Modified3DUnet model obtained from pykao/Modified-3D-UNet-Pytorch on Github.

# To specify:

# Paths where to load data from and save the models to
preprocessed_data_path = r'/home/artur-cmic/Desktop/UCL/Brats2019/Data/Preprocessed'
save_model_path = r'/home/artur-cmic/Desktop/UCL/Brats2019/Model_Saves_V4'
save_losses_path = r'/home/artur-cmic/Desktop/UCL/Brats2019/KFold_Losses_V4_new.txt'
dice_as_loss = True  # Whether to use dice as loss function, if False, MultiClass Crossentropy is used

# Training Parameters
batch_size = 2
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 5}
max_epochs = 150

# Model Parameters
in_channels = 4
n_classes = 4
base_n_filter = 16
n_folds = 5  # Number of folds in cross-validation

##############################################################################

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/training_V4')

if not os.path.isdir(save_model_path):
    os.mkdir(save_model_path)

# Use GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Get paths and names (IDS) of folders that store the multimodal training data
folder_paths = []
folder_ids = []
for subdir in os.listdir(preprocessed_data_path):
    folder_paths.append(os.path.join(preprocessed_data_path, subdir))
    folder_ids.append(subdir)

# Shuffle them around, keeping same seed to make sure same shuffling is used if training is interrupted and needs to be continued
random.seed(4)
random.shuffle(folder_paths)
random.seed(4)
random.shuffle(folder_ids)

# Setup KFold Cross Validation
kf = KFold(n_splits=n_folds, shuffle=False)  # Shuffle=false to get the same shuffling scheme every run
fold_nr = 1

# Training Loop
for fold in kf.split(folder_paths):
    iter_nr = 1
    train_idx = fold[0]
    valid_idx = fold[1]
    train_set = Dataset([folder_paths[i] for i in train_idx], [folder_ids[i] for i in train_idx])
    valid_set = Dataset([folder_paths[i] for i in valid_idx], [folder_ids[i] for i in valid_idx])
    train_loader = data.DataLoader(train_set, **params)
    valid_loader = data.DataLoader(valid_set, **params)

    # Model
    model = Modified3DUNet(in_channels, n_classes, base_n_filter)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # Loss and optimizer
    if not dice_as_loss:
        criterion = torch.nn.CrossEntropyLoss().to(device)  # For cross entropy
    optimizer = torch.optim.Adam(model.parameters())

    # Load model and optimizer parameters if the training was interrupted and must be continued - need to also change epoch range in for loop
    # checkpoint = torch.load("/content/drive/My Drive/Brats2019/Model_Saves_KFold/Fold_1_Epoch_30_Train_Loss_0.0140_Valid_Loss_0.0137.tar")
    # model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #model.train()
    for epoch in range(1, max_epochs + 1):
        start_time = time.time()
        train_losses = []
        for batch, labels in train_loader:

            # Data Augment
            #augmenter = DataAugment(batch,labels)
            #batch,labels = augmenter.augment()
            # Transfer batch and labels to GPU
            scans, masks = batch.to(device), labels.to(device)
            output, seg_layer = model(scans)

            if(iter_nr % 100 == 0):
                img = gen_plot(batch[0], labels[0], seg_layer[0])
                writer.add_image('Fold {}'.format(fold_nr), img, iter_nr)

            if dice_as_loss:
                masks = nn.functional.one_hot(masks.view(-1), num_classes=n_classes)
                train_loss = dice(output, masks)
            else:
                masks = masks.view(-1)
                train_loss = criterion(output, labels)  # For cross entropy

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())

            writer.add_scalar('Train Loss Fold {}'.format(fold_nr), train_loss.item(), iter_nr)
            iter_nr += 1

        # Get training loss after every epoch
        train_loss_ep = np.mean(train_losses)
        writer.add_scalar('Train Loss PerEpoch Fold {}'.format(fold_nr), train_loss_ep, epoch)
        # Get validation loss after every epoch
        valid_losses = []
        with torch.no_grad():
            for batch, labels in valid_loader:
                batch, labels = batch.to(device), labels.to(device)
                output, seg_layer = model(batch)
                if dice_as_loss:
                    masks = nn.functional.one_hot(masks.view(-1), num_classes=n_classes)
                    valid_loss = dice(output, masks)
                else:
                    masks = masks.view(-1)
                    train_loss = criterion(output, labels)  # For cross entropy
                valid_losses.append(valid_loss.item())
        valid_loss_ep = np.mean(valid_losses)
        writer.add_scalar('Valid Loss PerEpoch Fold {}'.format(fold_nr), valid_loss_ep, epoch)
        elapsed_time = time.time() - start_time

        # Save the model parameters
        if (epoch % 10 == 0):
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, "{}/Fold_{}_Epoch_{}.tar".format(save_model_path, fold_nr, epoch))
    fold_nr = fold_nr + 1