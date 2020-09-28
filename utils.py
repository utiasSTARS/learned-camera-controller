import torch
from torch import nn
from torchvision import transforms

import glob
import os.path
from PIL import Image

import pickle

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

def print_network(net):
	num_params = sum([param.numel() for param in net.parameters()])
	print(net)
	print('Total number of parameters: {}'.format(num_params))

def concatenate_dicts(*dicts):
	concat_dict = {} # Create a dictionary
	for key in dicts[0]: 
		concat_dict[key] = [] # loop through the keys in the first dictionary, assign to new dictionary
		for d in dicts:
			val = d[key] # loop through each dictionary, the val var is the value of the key in each of the dictionaries
			if isinstance(val, list):
				concat_dict[key] = concat_dict[key] + val # if val is a list, append to the new dictionary
			else:
				concat_dict[key].append(val) # if there is only once instance, add to dict

	return concat_dict

def compute_dict_avg(dict):
	avg_dict = {}
	for key, val in dict.items():
		avg_dict[key] = np.mean(np.array(val))
	return avg_dict

def tag_dict_keys(dict, tag):
	new_dict = {}
	for key, val in dict.items():
		new_key = key + '/' + tag
		new_dict[new_key] = val
	return new_dict

