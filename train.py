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

# Use GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Get paths and names (IDS) of folders that store the multimodal training data
folder_paths = []
folder_IDS = []
for grade in os.listdir(data_path):
    for subdir in os.listdir(os.path.join(data_path, grade)):
        folder_paths.append(os.path.join(data_path, grade, subdir))
        folder_IDS.append(subdir)
# Shuffle them around
random.seed(4)
random.shuffle(folder_paths)
random.seed(4)
random.shuffle(folder_IDS)

# Training Parameters
params = {'batch_size': 4,
          'shuffle': True,
          'num_workers': 5}
max_epochs = 100

# Model Parameters
in_channels = 4
n_classes = 4
base_n_filter = 16

# Training Loop
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=False)

fold_nr = 1
for fold in kf.split(folder_paths):
    train_idx = fold[0]
    valid_idx = fold[1]
    train_set = Dataset([folder_paths[i] for i in train_idx], [folder_IDS[i] for i in train_idx])
    valid_set = Dataset([folder_paths[i] for i in valid_idx], [folder_IDS[i] for i in valid_idx])
    train_loader = data.DataLoader(train_set, **params)
    valid_loader = data.DataLoader(valid_set)

    model = Modified3DUNet(in_channels, n_classes, base_n_filter).to(device)
    # model.load_state_dict(torch.load("/content/drive/My Drive/Brats2019/Model_Saves/Epoch_151_Loss_0.0094.pt"))
    # model.eval()

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(max_epochs):
        i = 1
        for local_batch, local_labels in train_loader:
            # Transfer batch and labels to GPU
            batch, labels = local_batch.to(device), local_labels.to(device)
            output, seg_layer = model(batch)

            train_loss = criterion(output, labels.view(-1))
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            # tbc.save_value('Training Loss', 'Fold {}'.format(fold_nr), epoch * len(train_set) + i, train_loss.item())
            if (i % 10 == 0):
                print('Fold [{}/{}], Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(fold_nr, n_folds, epoch + 1, max_epochs, i, len(train_set), train_loss.item()))
            i = i + 1
        # Get validation loss after every epoch
        with torch.no_grad():
            valid_losses = []
            for local_batch, local_labels in valid_loader:
                valid_batch, valid_labels = local_batch.to(device), local_labels.to(device)
                output, seg_layer = model(valid_batch)
                valid_loss = criterion(output, valid_labels.view(-1))
                valid_losses.append(valid_loss.item())
        # tbc.save_value('Validation Loss', 'Fold {}'.format(fold_nr), epoch + 1, np.mean(valid_losses))

        # Save the model parameters after every epoch
        torch.save(model.state_dict(), "{}/Fold_{}_Epoch_{}_Train_Loss_{:.4f}_Valid_Loss_{:.4f}.pt".format(save_model_path, fold_nr, epoch + 1, train_loss.item(), valid_loss.item()))
    fold_nr = fold_nr + 1
