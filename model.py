## Imports
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold


# Function to plot 3-dimensional torch tensors as images
def torch_img(tensor):
  to_pil = torchvision.transforms.ToPILImage()
  img = to_pil(tensor)
  plt.imshow(img)
  plt.show()


# Function to load custom dataset from image folder hierarchy
root = '/content/new_resamples/'

def load_dataset():
  train_dataset = torchvision.datasets.ImageFolder(
      root = root, 
      transform = torchvision.transforms.ToTensor()
  )
  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=64,
      num_workers=0,
      shuffle=True
  )
  return train_loader.dataset


# Function to split custom dataset into train and test sets
def get_split(loader, batch=30, prop=0.8):
  
  default_load = lambda data, batch=batch: torch.utils.data.DataLoader(
      data,
      batch_size=batch,
      num_workers=0,
      shuffle=False
  )

  train_ix = int(prop * len(loader))
  test_ix = len(loader) - train_ix
  train, test = torch.utils.data.random_split(loader, [train_ix, test_ix])
  return map(default_load, (train, test))


## Net class for Log-Mel Spectrograms
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    # LFLB Block 1
    self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
    self.bn1 = nn.BatchNorm2d(64)
    self.pool1 = nn.MaxPool2d(2,2)
    
    # LFLB Block 2
    self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
    self.bn2 = nn.BatchNorm2d(64)
    self.pool2 = nn.MaxPool2d(4,4)

    # LFLB Block 3
    self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
    self.bn3 = nn.BatchNorm2d(128)
    self.pool3 = nn.MaxPool2d(4,4)

    # LFLB Block 4
    self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
    self.bn4 = nn.BatchNorm2d(128)
    self.pool4 = nn.MaxPool2d(4,4)

    # LSTM
    self.lstm = nn.LSTM(input_size=128, hidden_size=256)

    # FC
    self.fc = nn.Linear(256, 7)

  def forward(self,x):
    # LFLB Block 1
    x = self.bn1(self.conv1(x))
    x = self.pool1(F.elu(x))

    # LFLB Block 2
    x = self.bn2(self.conv2(x))
    x = self.pool2(F.elu(x))

    # LFLB Block 3
    x = self.bn3(self.conv3(x))
    x = self.pool3(F.elu(x))

    # LFLB Block 4
    x = self.bn4(self.conv4(x))
    x = self.pool4(F.elu(x))

    # LSTM
    x = x.view(x.size()[0], 128, -1)
    x = x.transpose(1,2)
    x = x.transpose(0,1).contiguous()
    x, _ = self.lstm(x)
    x = x.squeeze()

    # FC
    x = self.fc(x)

    return F.log_softmax(x, dim=1)

net = Net()
print(net)


def train(mod, n_epochs=30, show=False, gpu=None):
	optimizer = optim.Adam(net.parameters(), lr=0.01)

	if gpu: 
		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		net.to(device)

	EPOCHS = n_epochs  # Number of epochs to train for
	n_batch = len(train_data)

	## Ensures step size that decays by factor of 0.1 every 10 epochs
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_batch*10)

	if show: 
		import time
		start = time.time()

	for epoch in range(EPOCHS):
	  if show: print('EPOCH:', epoch)
	  for i, data in enumerate(train_data):
	    X,y = data
	    if gpu: 
	    	X,y = X.to(device), y.to(device)
	    net.zero_grad()
	    output = net(X)
	    loss = F.nll_loss(output, y)
	    loss.backward()
	    optimizer.step()
	    scheduler.step()
	  if show:
	  	delta = time.time() - start
	  	print(delta)
	  	print(loss)

	return mod