import torch
import numpy as np
import random
import os
from torch.utils import data
from sklearn.model_selection import KFold
from data_loaders import Dataset
from data_augment import DataAugment
from models import Modified3DUNet

##############################################
# Code for training the Modified3DUnet model obtained from pykao/Modified-3D-UNet-Pytorch on Github.
##############################################

# Paths where to load data from and save the models to
preprocessed_data_path = r'/home/artur-cmic/Desktop/Brats2019/Data/Preprocessed'
save_model_path = r'/home/artur-cmic/Desktop/Brats2019/KFold_Validation_V2/Model_Saves'
save_losses_path = r'/home/artur-cmic/Desktop/Brats2019/KFold_Validation_V2'

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

# Training Parameters
batch_size = 2
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 5}
max_epochs = 100

# Model Parameters
in_channels = 4
n_classes = 4
base_n_filter = 16

# Setup KFold Cross Validation
n_folds = 5  # Number of folds in cross-validation
kf = KFold(n_splits=n_folds, shuffle=False)  # Shuffle=false to get the same shuffling scheme every run
fold_nr = 1

# Training Loop
for fold in kf.split(folder_paths):
    train_idx = fold[0]
    valid_idx = fold[1]
    train_set = Dataset([folder_paths[i] for i in train_idx], [folder_ids[i] for i in train_idx])
    valid_set = Dataset([folder_paths[i] for i in valid_idx], [folder_ids[i] for i in valid_idx])
    train_loader = data.DataLoader(train_set, **params)
    valid_loader = data.DataLoader(valid_set, **params)

    # Model
    model = Modified3DUNet(in_channels, n_classes, base_n_filter)
    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # Load model and optimizer parameters if the training was interrupted and must be continued - need to also change epoch range in for loop
    # checkpoint = torch.load("/content/drive/My Drive/Brats2019/Model_Saves_KFold/Fold_1_Epoch_30_Train_Loss_0.0140_Valid_Loss_0.0137.tar")
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.to(device)
    #model.train()

    for epoch in range(1, max_epochs + 1):
        train_losses = []
        for batch, labels in train_loader:
            # Data Augment
            augmenter = DataAugment(batch,labels)
            batch,labels = augmenter.augment()

            # Transfer batch and labels to GPU
            batch, labels = batch.to(device), labels.to(device)
            output, seg_layer = model(batch)
            train_loss = criterion(output, labels.view(-1))
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())

        # Get training loss after every epoch
        train_loss_ep = np.mean(train_losses)

        # Get validation loss after every epoch
        valid_losses = []
        with torch.no_grad():
            for batch, labels in valid_loader:
                batch, labels = batch.to(device), labels.to(device)
                output, seg_layer = model(batch)
                valid_loss = criterion(output, labels.view(-1))
                valid_losses.append(valid_loss.item())
        valid_loss_ep = np.mean(valid_losses)

        # Save the training and validation losses to file
        losses_file = open("{}/KFold_Losses.txt".format(save_losses_path), "a")
        losses_file.write("Fold_{}_Epoch_{}_TrainAvg_{:.4f}_ValidAvg_{:.4f}_TrainLast_{:.4f}_ValidLast_{:.4f}\n".format(fold_nr, epoch, train_loss_ep, valid_loss_ep, train_loss.item(), valid_loss.item()))
        losses_file.close()

        print('Fold [{}/{}], Epoch [{}/{}], Train Loss: {:.4f}, Valid Loss {:.4f}'.format(fold_nr, n_folds, epoch, max_epochs, train_loss_ep, valid_loss_ep))

        # Save the model parameters
        if (epoch % 1 == 0):
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                       "{}/Fold_{}_Epoch_{}.tar".format(save_model_path, fold_nr, epoch))

    fold_nr = fold_nr + 1