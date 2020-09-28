import torch
import torch.nn as nn
import torch.utils.data
import torchvision.models as models

import os
import utils
import numpy as np

import ipdb

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

class CNN_EG_SMALL(torch.nn.Module):
	def __init__(self): 
		super(CNN_EG_SMALL, self).__init__()
		
		### Initialize the various Network Layers
		self.layer1 = nn.Sequential(
			nn.Conv2d(15, 30, stride=1, kernel_size=(5,5)),
			nn.MaxPool2d((3,3),stride=3),
			nn.BatchNorm2d(30),
			nn.ReLU(),
			nn.Dropout2d(p=0.4)
		)

		self.layer2 = nn.Sequential(
			nn.Conv2d(30, 60, stride=1, kernel_size=(3,3)),
			nn.MaxPool2d((3,3),stride=3),
			nn.BatchNorm2d(60),
			nn.ReLU(),
			nn.Dropout2d(p=0.4)
		)

		self.layer3 = nn.Sequential(
			nn.Conv2d(60, 120, stride=1, kernel_size=(3,3)),
			nn.MaxPool2d((3,3),stride=3),
			nn.BatchNorm2d(120),
			nn.ReLU(),
			nn.Dropout2d(p=0.4)
		)

		self.layer4 = nn.Sequential(
			nn.Conv2d(120,240, kernel_size=(7,7)),
			nn.BatchNorm2d(240),
			nn.ReLU(),
			nn.Dropout2d(p=0.4)
		)

		self.fc = nn.Sequential(
			nn.Linear(240, 128),
			nn.BatchNorm1d(128),
			nn.ReLU(),
			nn.Dropout2d(p=0.3),
			nn.Linear(128, 64),
			nn.BatchNorm1d(64),
			nn.ReLU(),
			nn.Dropout2d(p=0.2),
			nn.Linear(64, 32),
			nn.BatchNorm1d(32),
			nn.Linear(32, 2)
		)       

	def forward(self, x):

		# print("Initial: ", x.shape)
		x = self.layer1(x)
		# print("Layer 1: ", x.shape)
		x = self.layer2(x)
		# print("Layer 2: ", x.shape)
		x = self.layer3(x)
		# print("Layer 3: ", x.shape)
		x = self.layer4(x)
		# print("Layer 4: ", x.shape)
		# x = self.layer5(x)
		x = x.view(-1, x.size(1))
		# print("Layer 5: ", x.shape)
		x = self.fc(x)
		# print("Final: ", x.shape)

		return x