import os
import os.path
import scipy.io as sio
import pickle
import glob

import numpy as np
import re
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import random
import datatransforms
from torchvision import transforms

import ipdb

class CustomDataset:
	""" Load and parse data from our collected datasets. Need to save this in a format where all image names and targets are stored as a dictionary."""

	def __init__(self, base_path, trajectory, labels, opts, **kwargs):
		self.base_path = base_path
		self.trajectory = trajectory
		self.labels = labels
		self.opts = opts
		self.sequence = opts.sequence
		self.parameter = opts.parameter
		self.training_data = {}
		self.validation_data = {}
		self.test_data = {}
		
	def __len__(self, targets):
		return len(targets)

	def generate_labels(self, label_files):
		"""
		We want to save the target from pose t+1 with images from t, t-1, t-2. First we need to obtain the target values from the name of the image file.

		Targets should be in the form of:
		
		"data_X_####_exp-X_gain-X.X.jpg"
		data_X - is the trajectory number
		#### - the number of the image in the trajectory - must be 4 characters (e.g., 0012)
		exp-X - is the target exposure
		gain-X.X - is the gain of the target (if a whole number, should have decimal zero - e.g., 3.0)
		"""
		targets = np.ndarray(shape=(len(label_files),3))

		for i, name in enumerate(label_files):
			params = [int(num) for num in re.findall(r'\d+', name)]
			gain = round(params[3] + params[4]/100, 2)
			targets[i,:] = (params[1], params[2], gain) # pose, exp, gain
			
			# print("The targets from generate_labels: \n", targets[i,:])

		return targets

	def read_params(self, filename):
		""" Read the parameter values from the filename and save them as integers"""
		params = re.findall(r'\d+', filename)
		gain = round(int(params[3]) + int(params[4])/100, 2)
		pose, exp = int(params[1]), int(params[2])

		# print('Pose: {}, exp: {}, gain: {}'.format(pose, exp, gain))
		# ipdb.set_trace()

		return pose, exp, gain

	def create_set(self):
		"""
		Loop through images in current folder and generate training pairs (sets) for each image and corresponding label.
		"""

		print("Creating dataset for moving data...")	
		key_name_num_train = 0
		key_name_num_test = 0
		key_name_num_val = 0

		val_indeces = self.opts.validation_sets

		for i, traj in enumerate(self.trajectory):
			print('Currently on: {}'.format(traj))

			if i in val_indeces:
				print('Generating data for {} as validation data...'.format(traj))
			else:
				print('Generating data for {} as training data...'.format(traj))
			
			targets = self.generate_labels(self.labels[i]) # target = len(label[i]) x 3

			cams = next(os.walk(os.path.join(self.base_path, traj)))[1]
			images_cam1 = next(os.walk(os.path.join(self.base_path, traj, cams[0])))[2]
			images_cam2 = next(os.walk(os.path.join(self.base_path, traj, cams[1])))[2]
			images_cam1.sort()
			images_cam2.sort()

			num_poses = self.__len__(targets)
			for j in range(num_poses):

				# Generate every possible training sample for this pose
				image_t_cam1 = os.path.join(self.base_path, traj, cams[0], images_cam1[j+2])
				image_t_cam2 = os.path.join(self.base_path, traj, cams[1], images_cam2[j+2])
				image_t1_cam1 = os.path.join(self.base_path, traj, cams[0], images_cam1[j+1])
				image_t1_cam2 = os.path.join(self.base_path, traj, cams[1], images_cam2[j+1])
				image_t2_cam1 = os.path.join(self.base_path, traj, cams[0], images_cam1[j])
				image_t2_cam2 = os.path.join(self.base_path, traj, cams[1], images_cam2[j])

				combinations = [[image_t_cam1, image_t1_cam1, image_t2_cam1],
								[image_t_cam1, image_t1_cam1, image_t2_cam2],
								[image_t_cam1, image_t1_cam2, image_t2_cam1],
								[image_t_cam1, image_t1_cam2, image_t2_cam2],
								[image_t_cam2, image_t1_cam1, image_t2_cam1],
								[image_t_cam2, image_t1_cam1, image_t2_cam2],
								[image_t_cam2, image_t1_cam2, image_t2_cam1],
								[image_t_cam2, image_t1_cam2, image_t2_cam2]]
				
				for combo in combinations:
					img_name_t = combo[0]
					img_name_t1 = combo[1]
					img_name_t2 = combo[2]

					# create a list of rows where each row is a training example
					example = []

					_, exp_t, gain_t = self.read_params(img_name_t)
					current_params_t = [exp_t, gain_t]

					_, exp_t1, gain_t1 = self.read_params(img_name_t1)
					current_params_t1 = [exp_t1, gain_t1]

					_, exp_t2, gain_t2 = self.read_params(img_name_t2)
					current_params_t2 = [exp_t2, gain_t2]
					
					if j < num_poses-2:
						targets_t = [targets[j,1], targets[j+1,1], targets[j+2,1], targets[j,2], targets[j+1,2], targets[j+2,2]]
					elif j < num_poses-1:
						targets_t = [targets[j,1], targets[j+1,1], targets[j+1,1], targets[j,2], targets[j+1,2], targets[j+1,2]]
					else:
						targets_t = [targets[j,1], targets[j,1], targets[j,1], targets[j,2], targets[j,2], targets[j,2]]	
					
					example = [img_name_t, img_name_t1, img_name_t2, current_params_t[0], current_params_t[1], current_params_t1[0], current_params_t1[1], current_params_t2[0], current_params_t2[1]]
					example.extend(targets_t)

					""" Example should be of the form:
					ex[0] = img1 name   ex[1] = img2 name   ex[2] = img3 name
					ex[3] = img1 curr exp   ex[4] = img1 curr gain
					ex[5] = img2 curr exp   ex[6] = img2 curr gain
					ex[7] = img3 curr exp   ex[8] = img3 curr gain
					ex[9] = cur tar exp     ex[10] = t+1 tar exp
					ex[11] = t+2 tar exp    ex[12] = cur tar gain
					ex[13] = t+1 tar gain   ex[14] = t+2 tar gain
					"""	

					# Generate placeholder key name based on the current number of examples
					if i in val_indeces:
						key_name = '{:06d}'.format(key_name_num_val)
						self.validation_data[key_name] = example
						key_name_num_val += 1
					
					else:
						key_name = '{:06d}'.format(key_name_num_train)
						self.training_data[key_name] = example
						key_name_num_train += 1


	def save_dataset(self):
		filename_train = self.opts.datadir + 'training_data/trainingdata.mat'
		filename_eval = self.opts.datadir + 'training_data/evaldata.mat'
		filename_test = self.opts.datadir + 'training_data/testdata.mat'
		os.makedirs(os.path.dirname(filename_train), exist_ok=True)
		sio.savemat(filename_train, self.training_data)
		sio.savemat(filename_eval, self.validation_data)
		sio.savemat(filename_test, self.test_data)
		return filename_train, filename_eval, filename_test

class TorchDataset(torch.utils.data.Dataset):
	""" Dataset for use in pytorch"""

	def __init__(self, mat_file, opts, transform=None):
		""" Args:
				targets (string): Path to file containing optimal images
				transform (optional, callable): Optional transform to be applied on a sample
		"""
		self.dataset = sio.loadmat(mat_file)
		self.opts = opts
		self.transform = transform
		self.sequence = opts.sequence
		self.image_info = list(self.dataset.values())[3:]

	def __len__(self):
		return len(self.image_info)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		img_name_t = self.image_info[idx][0].rstrip()
		img_name_t1 = self.image_info[idx][1].rstrip()
		img_name_t2 = self.image_info[idx][2].rstrip()
		img_t = Image.open(img_name_t)
		img_t1 = Image.open(img_name_t1)
		img_t2 = Image.open(img_name_t2)
		target = self.image_info[idx][3:].astype(float).reshape(1,-1)
		sample = {'image': [img_t, img_t1, img_t2], 'target': target}

		if self.transform:
			sample = self.transform(sample)

		return sample