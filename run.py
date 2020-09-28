import torch
from torch.utils.data import DataLoader

import numpy as np
from numpy import random
import torchvision
from torchvision import datasets, models, transforms

from dataset import CustomDataset, TorchDataset
from options import Options
import model
import experiment
import datatransforms
import utils

import os
import time
import pickle
import argparse
from pathlib import Path

import ipdb

### COMMAND LINE ARGUEMNTS ###
parser = argparse.ArgumentParser()
parser.add_argument('stage', type=str, choices=['train', 'test', 'both'])
parser.add_argument('method', type=str, choices=['features', 'gridsearch', 'hybrid'])
parser.add_argument('dataPath', type=str)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--hyperparameter', action='store_true')
parser.add_argument('--crossValidation', action='store_true')
args = parser.parse_args()

resume_from_epoch = 'latest' if args.resume else None

### CONFIGURATION ###
opts = Options()

# Setting options
opts.method = args.method
opts.experiment_name = opts.method+'/'
opts.hyperparameter = True if args.hyperparameter else False
opts.cross_validation = True if args.crossValidation else False
opts.datapath = args.dataPath
	
# List the names of the trajectories in the dataset
base_path = opts.datapath
trajectory = next(os.walk(base_path))[1]
trajectory.sort()
num_trajectories = len(trajectory)

# Testing stuff, remove ##############################################################
trajectory = trajectory[39:41]
print(trajectory)

# Get lists of the target images
optimal_images = []
for traj in trajectory:
	if opts.method == 'features' or opts.method == 'gridsearch':
		file_path = 'data/'+opts.method+'/optimal_images_orb_'+traj+'.pckl'
	elif opts.method == 'hybrid':
		filepath = 'data/'+opts.method+'hybrid_equal/optimal_images_orb_'+traj+'.pckl'
		# filepath = 'data/'+opts.method+'hybrid_gridweighted/optimal_images_orb_'+traj+'.pckl'
	optimal_images.append(pickle.load(open(file_path, 'rb')))

# Check if dataset exists already - if not, create it
if os.path.exists(opts.datadir + 'training_data/trainingdata.mat'):
	print('\n*** Training and test .mat data already exist. ***\n')
	mat_file_train = opts.datadir + 'training_data/trainingdata.mat'
	mat_file_test = opts.datadir + 'training_data/testdata.mat'
	mat_file_val = opts.datadir + 'training_data/evaldata.mat'
	pass
else:
	print('\n*** Creating training, validation, and test .mat data. ***\n')
	dataset = CustomDataset(base_path, trajectory, optimal_images, opts)
	dataset.create_set()
	mat_file_train, mat_file_val, mat_file_test = dataset.save_dataset()

# HYPERPARAMETER TUNING
if opts.cross_validation == False:

	if opts.hyperparameter == True:

		# grid of hyperparameter values
		param_grid = {
			'lr': [1e-5, 1e-4, 1e-3],
			'batch_size': [64],
			'loss_type': ['l1', 'mse'],
			'loss_formulation': ['percent_delta', 'additive_delta', 'multiplicative_delta'],
			# 'loss_formulation': ['absolute'],
			'image_normalization': [False],
			'compensated_target': [True, False],
		}

		all_params = []
		best_params = []

		# Randomly search the hyperparameter space
		opts.jobs = 1
		num_models = 12
		for i in range(num_models):
			print('\n\n{:-^50}'.format(' Beginning Job '))
			print('Starting Job {} of {}'.format(opts.jobs, num_models))
			print('-' * 50 + '\n')

			# Generate a randomly selected trial of hyperparameters
			random.seed(60)
			random_params = {k: random.choice(v) for k, v in param_grid.items()}

			# Create a list of combinations
			curr_params = [float(random_params['lr']), int(random_params['batch_size']), random_params['loss_type'], random_params['loss_formulation'], random_params['image_normalization'], random_params['compensated_target']]

			# if first set of params, initialize the global list
			if not all_params:
				all_params.append(curr_params)

			else:
				# check if combination has been seen before, resample if so
				while curr_params in all_params:
					random_params = {k: random.choice(v) for k, v in param_grid.items()}
					curr_params = [float(random_params['lr']), int(random_params['batch_size']), random_params['loss_type'], random_params['loss_formulation'], random_params['image_normalization'], random_params['compensated_target']]

				all_params.append(curr_params)

			opts.lr = curr_params[0]
			opts.batch_size = curr_params[1]
			opts.loss_type = curr_params[2]
			opts.loss_formulation = curr_params[3]
			opts.image_normalization = curr_params[4]
			opts.compensated_target = curr_params[5]

			# Create data transforms
			if opts.image_normalization == True:
				data_transforms = transforms.Compose([
						datatransforms.HorizontalFlip(opts),
						datatransforms.VerticalFlip(opts),
						datatransforms.ToTensor(opts),
						datatransforms.NormalizeImage(opts)
						])
			elif opts.image_normalization == False:
				data_transforms = transforms.Compose([
						datatransforms.HorizontalFlip(opts),
						datatransforms.VerticalFlip(opts),
						datatransforms.ToTensor(opts),
						])
			training_dataset = TorchDataset(mat_file_train, opts, data_transforms)
			validation_dataset = TorchDataset(mat_file_val, opts, data_transforms)
			testing_dataset = TorchDataset(mat_file_test, opts, data_transforms)

			print('Running {} training...'.format(opts.parameter))

			model = model.CNN_EG_SMALL()
			model.to(opts.device)

			print('\n{:-^50}'.format(' Network Initialized '))
			utils.print_network(model)
			print('-' * 50 + '\n')

			if args.stage == 'train' or args.stage == 'both':
				print(opts)
				experiment.train(opts, model, training_dataset, validation_dataset, opts.train_epochs, resume_from_epoch=resume_from_epoch)

			if args.stage == 'test' or args.stage == 'both':
				print('\n{:-^50}'.format(' Testing Model '))
				experiment.test(opts, model, testing_dataset, save_loss=True)

		opts.jobs += 1

	if opts.hyperparameter == False:

		# Selected hyperparameters
		param_grid = {
		'lr': 1e-4,
		'batch_size': 4,
		'loss_type': 'l1',
		'loss_formulation': 'absolute',
		'image_normalization': False,
		'compensated_target': False,
		}

		# Assigning the hyperparameters
		opts.lr = param_grid['lr']
		opts.batch_size = param_grid['batch_size']
		opts.loss_type = param_grid['loss_type']
		opts.loss_formulation = param_grid['loss_formulation']
		opts.image_normalization = param_grid['image_normalization']
		opts.compensated_target = param_grid['compensated_target']
		print(opts)

		# Create data transforms
		data_transforms = transforms.Compose([
						datatransforms.HorizontalFlip(opts),
						datatransforms.VerticalFlip(opts),
						datatransforms.ToTensor(opts),
						])
		training_dataset = TorchDataset(mat_file_train, opts, data_transforms)
		validation_dataset = TorchDataset(mat_file_val, opts, data_transforms)
		testing_dataset = TorchDataset(mat_file_test, opts, data_transforms)

		print('Running {} training...'.format(opts.parameter))
		model = model.CNN_EG_SMALL()
		model.to(opts.device)

		print('\n{:-^50}'.format(' Network Initialized '))
		utils.print_network(model)
		print('-' * 50 + '\n')

		if args.stage == 'train' or args.stage == 'both':
			print(opts)
			experiment.train(opts, model, training_dataset, validation_dataset, opts.train_epochs, resume_from_epoch=resume_from_epoch)

		if args.stage == 'test' or args.stage == 'both':
			print('\n{:-^50}'.format(' Testing Model '))
			experiment.test(opts, model, training_dataset, save_loss=True)

if opts.cross_validation == True:
	print('*** Training with 5-fold Cross Validation ***\n')

	# Selected hyperparameters
	param_grid = {
	'lr': 1e-4,
	'batch_size': 64,
	'loss_type': 'l1',
	'loss_formulation': 'absolute',
	'image_normalization': False,
	'compensated_target': False
	}

	# Assigning the hyperparameters
	opts.lr = param_grid['lr']
	opts.batch_size = param_grid['batch_size']
	opts.loss_type = param_grid['loss_type']
	opts.loss_formulation = param_grid['loss_formulation']
	opts.image_normalization = param_grid['image_normalization']
	opts.compensated_target = param_grid['compensated_target']
	print(opts)

	print('Running {} training...'.format(opts.parameter))

	# Perform 5-fold cross validation
	past_idx = []
	for i in range(5):

		# Randomly select two combinations of validation sets:
		if i == 0:
			val_index_1 = np.random.randint(num_trajectories)
			past_idx.append(val_index_1)
			val_index_2 = np.random.randint(num_trajectories)
			while val_index_2 in past_idx:
				val_index_2 = np.random.randint(num_trajectories)
			past_idx.append(val_index_2)
			val_index_3 = np.random.randint(num_trajectories)
			while val_index_3 in past_idx:
				val_index_3 = np.random.randint(num_trajectories)
			past_idx.append(val_index_3)
		else:
			val_index_1 = np.random.randint(num_trajectories)
			while val_index_1 in past_idx:
				val_index_1 = np.random.randint(num_trajectories)
			past_idx.append(val_index_1)
			val_index_2 = np.random.randint(num_trajectories)
			while val_index_2 in past_idx:
				val_index_2 = np.random.randint(num_trajectories)
			past_idx.append(val_index_2)
			val_index_3 = np.random.randint(num_trajectories)
			while val_index_3 in past_idx:
				val_index_3 = np.random.randint(num_trajectories)
			past_idx.append(val_index_3)
		
		opts.validation_sets = [val_index_1, val_index_2, val_index_3]
		# Create our dataset with different validation set
		print('\n*** Creating training, validation, and test .mat data. ***\n')
		dataset = CustomDataset(base_path, trajectory, optimal_images, opts)
		dataset.create_set()
		mat_file_train, mat_file_val, mat_file_test = dataset.save_dataset()

		# Create data transforms
		if opts.image_normalization == True:
			data_transforms = transforms.Compose([
					datatransforms.HorizontalFlip(opts),
					datatransforms.VerticalFlip(opts),
					datatransforms.ToTensor(opts),
					datatransforms.NormalizeImage(opts)
					])
		elif opts.image_normalization == False:
			data_transforms = transforms.Compose([
					datatransforms.HorizontalFlip(opts),
					datatransforms.VerticalFlip(opts),
					datatransforms.ToTensor(opts),
					])
		training_dataset = TorchDataset(mat_file_train, opts, data_transforms)
		validation_dataset = TorchDataset(mat_file_val, opts, data_transforms)
		testing_dataset = TorchDataset(mat_file_test, opts, data_transforms)

		# Create our own model for each k-fold
		model = model.CNN_EG_SMALL()
		model.to(opts.device)

		print('\n{:-^50}'.format(' Network Initialized '))
		utils.print_network(model)
		print('-' * 50 + '\n')

		if args.stage == 'train' or args.stage == 'both':
			print(opts)
			experiment.train(opts, model, training_dataset, validation_dataset, opts.train_epochs, resume_from_epoch=resume_from_epoch)

		if args.stage == 'test' or args.stage == 'both':
			print('\n{:-^50}'.format(' Testing Model '))
			experiment.test(opts, model, training_dataset, save_loss=True)

print('-' * 50 + '\n')
print('Training Complete!')
print('-' * 50 + '\n')