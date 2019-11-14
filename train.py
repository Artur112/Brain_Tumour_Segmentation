import torch
import numpy as np
import random
import os
from torch.utils import data
from sklearn.model_selection import KFold
from data_loaders import Dataset
from models import Modified3DUNet

# Code was run in google colab before, so code for tensorboard must be added

# Paths where to load data from and save the models to
data_path = 'xxxx'
save_model_path = 'xxxx'
save_losses_path = 'xxxx'

# tbc=TensorBoardColab() #For monitoring training losses in tensorboard

# Use GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Get paths and names (IDS) of folders that store the multimodal training data
data_path = 'drive/My Drive/Brats2019/Training_Data/'
folder_paths = []
folder_IDS = []
for grade in os.listdir(data_path):
    for subdir in os.listdir(os.path.join(data_path, grade)):
        folder_paths.append(os.path.join(data_path, grade, subdir))
        folder_IDS.append(subdir)
# Shuffle them around, keeping same seed to make sure same shuffling is used if training is interrupted and needs to be continued
random.seed(4)
random.shuffle(folder_paths)
random.seed(4)
random.shuffle(folder_IDS)

# Training Parameters
batch_size = 4
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 5}
max_epochs = 100

# Model Parameters
in_channels = 4
n_classes = 4
base_n_filter = 16

# Training Loop
n_folds = 5  # Number of folds in cross-validation
kf = KFold(n_splits=n_folds, shuffle=False)  # Shuffle=false to get the same shuffling scheme every run
fold_nr = 1
for fold in kf.split(folder_paths):
    # if(fold_nr>1):
    train_idx = fold[0]
    valid_idx = fold[1]
    train_set = Dataset([folder_paths[i] for i in train_idx], [folder_IDS[i] for i in train_idx])
    valid_set = Dataset([folder_paths[i] for i in valid_idx], [folder_IDS[i] for i in valid_idx])
    train_loader = data.DataLoader(train_set, **params)
    valid_loader = data.DataLoader(valid_set)

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
    # model.train()

    for epoch in range(1, max_epochs + 1):
        train_losses = []
        for batch, labels in train_loader:
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
        # tbc.save_value('Training Loss', 'Fold {}'.format(fold_nr), epoch, train_loss_ep)

        # Get validation loss after every epoch
        valid_losses = []
        with torch.no_grad():
            for batch, labels in valid_loader:
                batch, labels = batch.to(device), labels.to(device)
                output, seg_layer = model(batch)
                valid_loss = criterion(output, labels.view(-1))
                valid_losses.append(valid_loss.item())
        valid_loss_ep = np.mean(valid_losses)
        # tbc.save_value('Validation Loss', 'Fold {}'.format(fold_nr), epoch , valid_loss_ep)

        # Save the training and validation losses to file
        losses_file = open("{}/KFold_Losses.txt".format(save_losses_path), "a")
        losses_file.write("Fold_{}_Epoch_{}_Train_{:.4f}_Valid_{:.4f}\n".format(fold_nr, epoch, train_loss_ep, valid_loss_ep))
        losses_file.close()

        print('Fold [{}/{}], Epoch [{}/{}], Train Loss: {:.4f}, Valid Loss {:.4f}'.format(fold_nr, n_folds, epoch, max_epochs, train_loss_ep, valid_loss_ep))

        # Save the model parameters
        if (epoch % 5 == 0):
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                       "{}/Fold_{}_Epoch_{}_Train-Loss_{:.4f}_Valid-Loss_{:.4f}.tar".format(save_model_path, fold_nr, epoch, train_loss_ep, valid_loss_ep))

    fold_nr = fold_nr + 1