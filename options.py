import numpy as np
import torch
import os
import re
import ipdb

class Options:
	"""
	Container class to store configuration parameters and common defaults
	"""

	def __init__(self):
		self.datadir = 'data/'
		self.results_dir = 'results/'
		self.experiment_name = 'name'
		self.datapath = 'default'
		self.parameter = 'exp+gain'
		self.sequence = 'multi_image'
		self.method = 'default'
		self.hyperparameter = False
		self.param_norm = False
		self.cross_validation = False

		self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
		self.dataloader_workers = 0

		# Hyperparameters
		self.train_epochs = 1
		self.lr = 1e-3
		self.batch_size = 12
		self.loss_type = 'default'
		self.loss_formulation = 'default'
		self.image_normalization = False
		self.compensated_target = False
		self.epsilon = 0.5

		self.checkpoint_interval = 10 # epochs
		self.jobs = 1
		self.best_model = 1e12
		self.validation_sets = [1]

		# Camera Settings
		self.max_exposure = 30000
		self.min_exposure = 75
		self.max_gain = 30

	def save_txt(self, name, *args, **kwargs):
		run = 0
		for ar in args:
			run = ar

		if self.hyperparameter == True:
			if run == 0:
				save_dir = os.path.join(self.results_dir, 'hyperparameter_tuning')
			else:
				save_dir = os.path.join(self.results_dir, 'hyperparameter_tuning', 'validation', run)
			os.makedirs(save_dir, exist_ok = True)
			save_file = os.path.join(save_dir, name)
			
		elif self.hyperparameter == False:
			if run == 0:
				save_dir = os.path.join(self.results_dir, self.experiment_name)
			else:
				save_dir = os.path.join(self.results_dir, self.experiment_name,'validation', run)
			os.makedirs(save_dir, exist_ok = True)
			save_file = os.path.join(save_dir, name)

		print("Saving config to {}".format(save_file))
		with open(save_file, 'wt') as file:
			file.write(self.__repr__())

	def load_opts(self, opts_file):
		""" 
		Load options from a saved txt file
		"""
		print('Loading options from \'{}\'.'.format(opts_file))
		args = vars(self)
		file = open(opts_file, "rt")
		for line in file.readlines():
			# check to see which key from args exists in this line
			for key in args.keys():
				if line.find(key) == True:
					val = line.split(": ",1)[1].strip()
					val_type = type(args[key])
					if isinstance(args[key], list):
						match1 = re.search(r'.*?\[(.*),.*', val)
						match2 = re.search(r'.*?, (.*)\]', val)
						val1 = match1.group(1)
						val2 = match2.group(1)
						val = [float(val1), float(val2)]
					else:
						val = val_type(val)
					args[key] = val

	def __repr__(self):
		""" This prints the string representation of the Options class """

		args = vars(self)
		string = '\n{:-^50}\n'.format(' Options ')
		for key, val in sorted(args.items()):
			string += ' {:25}: {}\n'.format(str(key), str(val))
		string += '-'* 50 + '\n'
		return string
