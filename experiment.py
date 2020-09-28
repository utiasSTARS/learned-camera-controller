import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import numpy as np
import torchvision
import progress.bar
import utils
import math

from torchvision import datasets, models, transforms
import model
import gc

import os
import time
import ipdb

def set_mode(mode, model):
	""" Set the network to train/eval mode. Affects the dropout and batchnorm. """
	if mode == 'train':
		model.train()
		print('\n{:-^50}'.format(' Network Mode '))
		print("Network now in '{}' mode.".format(mode))
		print('-' * 50 + '\n')
	elif mode == 'eval':
		model.eval()
		print('\n{:-^50}'.format(' Network Mode '))
		print("Network now in '{}' mode.".format(mode))
		print('-' * 50 + '\n')
	else:
		raise ValueError(
			"Invalid mode '{}'. Valid options are 'train' and 'eval'.".format(mode))

def set_data(data, opts):
	""" Set the input tensors. """

	device = torch.device(opts.device)

	image_data = data['target'].float().to(device)
		
	expt = image_data[:, :,0]
	expt1 = image_data[:, :, 2]
	expt2 = image_data[:, :, 4]

	gaint = image_data[:, :, 1]
	gaint1 = image_data[:, :, 3]
	gaint2 = image_data[:, :, 5]

	adjusted_expt = ((expt-opts.min_exposure)/(opts.max_exposure-opts.min_exposure))*(255)
	adjusted_expt1 = ((expt1-opts.min_exposure)/(opts.max_exposure-opts.min_exposure))*(255)
	adjusted_expt2 = ((expt2-opts.min_exposure)/(opts.max_exposure-opts.min_exposure))*(255)

	adjusted_gaint = (gaint/opts.max_gain)*(255)
	adjusted_gaint1 = (gaint1/opts.max_gain)*(255)
	adjusted_gaint2 = (gaint2/opts.max_gain)*(255)

	template = torch.ones((len(expt), 1, 224,224)).to(device)
	adjusted_expt_new = adjusted_expt[:,:,None, None]
	adjusted_expt1_new = adjusted_expt1[:,:,None, None]
	adjusted_expt2_new = adjusted_expt2[:,:,None, None]
	adjusted_gaint_new = adjusted_gaint[:,:,None, None]
	adjusted_gaint1_new = adjusted_gaint1[:,:,None, None]
	adjusted_gaint2_new = adjusted_gaint2[:,:,None, None]
	exp_t = template*adjusted_expt_new.expand_as(template)
	exp_t1 = template*adjusted_expt1_new.expand_as(template)
	exp_t2 = template*adjusted_expt2_new.expand_as(template)
	gain_t = template*adjusted_gaint_new.expand_as(template)
	gain_t1 = template*adjusted_gaint1_new.expand_as(template)
	gain_t2 = template*adjusted_gaint2_new.expand_as(template)

	# del template

	parameter_t = torch.cat((exp_t, gain_t), 1)
	parameter_t1 = torch.cat((exp_t1, gain_t1), 1)
	parameter_t2 = torch.cat((exp_t2, gain_t2), 1)

	images = data['image']

	img1, img2, img3 = images[0].to(device), images[1].to(device), images[2].to(device)

	new_img1 = torch.cat((img1, parameter_t), 1)
	new_img2 = torch.cat((img2, parameter_t1), 1)
	new_img3 = torch.cat((img3, parameter_t2), 1)

	im1_im2 = torch.cat((new_img1, new_img2), 1)

	images = torch.cat((im1_im2, new_img3),1)

	# del im1_im2, new_img1, new_img2, new_img3, img1, img2, img3

	return images, image_data

def optimize(model, opt, loss_function, opts, image, image_data):
	""" Do one step of training with the current input tensors. """
	opt.zero_grad()
	loss_p, _ = forward(opts, model, loss_function, image, image_data)
	loss_p.backward()
	opt.step()

	return loss_p

def forward(opts, model, loss_function, image, image_data, compute_loss=True):
	""" Evaluate the forward pass of the parameter estimator model. """
	
	delta_param = model.forward(image)
	delta_exp = delta_param[:,0].unsqueeze(1)
	delta_gain = delta_param[:,1].unsqueeze(1)	
	loss_p = None

	if compute_loss:
		if opts.compensated_target == True:
			target_param_e = ((((image_data[:,:,6] + image_data[:,:,7] + image_data[:,:,8])/3)-opts.min_exposure)/(opts.max_exposure-opts.min_exposure))*(1)
			target_param_g = (((image_data[:,:,9] + image_data[:,:,10] + image_data[:,:,11])/3)/opts.max_gain)*(1)

		elif opts.compensated_target == False:
			target_param_e = ((image_data[:,:,6]-opts.min_exposure)/(opts.max_exposure-opts.min_exposure))*(1)
			target_param_g = (image_data[:,:,9]/opts.max_gain)*(1)
		
		loss_p_e = loss_function(delta_exp, target_param_e)
		loss_p_g = loss_function(delta_gain, target_param_g)
		loss_p = (1-opts.epsilon)*loss_p_e + opts.epsilon*loss_p_g
	
	return loss_p, delta_param

def model_test(opts, model, loss_function, image, image_data, compute_loss):
	""" Evaluate the model and test loss without optimizing. """
	with torch.no_grad():
		loss_p, delta_param = forward(opts, model, loss_function, image, image_data, compute_loss)

		return loss_p, delta_param

def get_errors(loss_p):
	""" Return a dictionary of the current errors. """
	error_dict = {'Loss_p': loss_p.item()}

	return error_dict

def get_data(opts, image, image_data, delta_param):
	"""
	Return a dictionary of relevant values for debugging purposes.
	"""
	current_exp = image_data[:,:,0]
	current_gain = image_data[:,:,1]
	target_exp = image_data[:,:,6]
	target_gain = image_data[:,:,9]
	delta_exp = delta_param[:,0]
	delta_gain = delta_param[:,1]

	# Calculate error and parameter estimate based on loss formulation
	predicted_exp = delta_exp
	predicted_gain = delta_gain

	error_exp = target_exp - predicted_exp
	error_gain =  target_gain - predicted_gain
	data_dict = {
		'experiment': opts.parameter,
		'target_exp': target_exp.cpu().squeeze().numpy().tolist(),
		'target_gain': target_gain.cpu().squeeze().numpy().tolist(),
		'current_exp': current_exp.cpu().squeeze().numpy().tolist(),
		'current_gain': current_gain.cpu().squeeze().numpy().tolist(),
		'delta_exp': delta_exp.cpu().squeeze().numpy().tolist(),
		'delta_exp': delta_gain.cpu().squeeze().numpy().tolist(),
		'predicted_exp': predicted_exp.cpu().squeeze().numpy().tolist(),
		'predicted_gain': predicted_gain.cpu().squeeze().numpy().tolist(),
		'difference/error_exp': error_exp.cpu().squeeze().numpy().tolist(),
		'difference/error_gain': error_gain.cpu().squeeze().numpy().tolist(),
	}
	return data_dict

def save_checkpoint(epoch, label, opts, model):
	""" Save model to file. """
	cur_dir = os.getcwd()
	model_dir = os.path.join(cur_dir, opts.results_dir, opts.experiment_name, 'checkpoints')
	os.makedirs(model_dir, exist_ok=True)
	model_file = os.path.join(model_dir, '{}_net.pth.tar'.format(label))

	model_dict = {'epoch': epoch,
					'label': label,
					'state_dict': model.state_dict()}

	print("Saving model to {}".format(model_file))
	torch.save(model_dict, model_file)

def load_checkpoint(opts, model, label):
	""" Load a model from a file. """
	cur_dir = os.getcwd()
	model_dir = os.path.join(cur_dir, opts.results_dir, opts.experiment_name, 'checkpoints')
	model_file = os.path.join(model_dir, '{}_net.pth.tar'.format(label))

	print("Loading model from {}".format(model_file))
	model_dict = torch.load(model_file, map_location=opts.device)

	device = torch.device(opts.device)
	model.to(opts.device)
	model.load_state_dict(model_dict['state_dict'])

	return model

def train(opts, model, train_data, val_data, num_epochs, resume_from_epoch=None):
	train_loader = DataLoader(train_data,
		                      batch_size=opts.batch_size,
		                      shuffle=True,
		                      num_workers=opts.dataloader_workers,
		                      pin_memory=True)
	val_loader = DataLoader(val_data,
		                    batch_size=opts.batch_size,
		                    shuffle=False,
		                    num_workers=opts.dataloader_workers,
		                    pin_memory=True)

	if opts.hyperparameter == False:
		if os.path.exists(os.path.join(opts.results_dir, opts.experiment_name, 'training')):
			previous_runs = os.listdir(os.path.join(opts.results_dir, opts.experiment_name, 'training'))
			if len(previous_runs) == 0:
				run_number = 1	
			else:
				run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1
		else:
			run_number = 1
	elif opts.hyperparameter == True:
		if os.path.exists(os.path.join(opts.results_dir, 'hyperparameter_tuning', 'training')):
			previous_runs = os.listdir(os.path.join(opts.results_dir, 'hyperparameter_tuning', 'training'))
			if len(previous_runs) == 0:
				run_number = 1	
			else:
				run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1
		else:
			run_number = 1

	log_dir_num = 'run_%02d' % run_number
	print("Currently on run #: ", run_number)
	log_learning_rate = 'lr_{}'.format(opts.lr)
	log_batch_size = 'batch_{}'.format(opts.batch_size)
	log_loss_type = '{}'.format(opts.loss_type)
	log_loss_formulation = '{}'.format(opts.loss_formulation)
	log_method = '{}'.format(opts.method)
	log_normalization = 'norm_{}'.format(opts.image_normalization)
	log_compensating = 'comp_{}'.format(opts.compensated_target)

	if opts.hyperparameter == False:
		train_log_dir = os.path.join(opts.results_dir, opts.experiment_name, 'training', log_dir_num, log_learning_rate, log_batch_size, log_loss_type, log_loss_formulation, log_method, log_normalization, log_compensating)
		val_log_dir = os.path.join(opts.results_dir, opts.experiment_name, 'validation', log_dir_num, log_learning_rate, log_batch_size, log_loss_type, log_loss_formulation, log_method, log_normalization, log_compensating)
		train_writer = SummaryWriter(train_log_dir)
		val_writer = SummaryWriter(val_log_dir)

	elif opts.hyperparameter == True:
		train_log_dir = os.path.join(opts.results_dir, 'hyperparameter_tuning', 'training', log_dir_num, log_learning_rate, log_batch_size, log_loss_type, log_loss_formulation, log_method, log_normalization, log_compensating)
		val_log_dir = os.path.join(opts.results_dir, 'hyperparameter_tuning', 'validation', log_dir_num, log_learning_rate, log_batch_size, log_loss_type, log_loss_formulation, log_method, log_normalization, log_compensating)
		train_writer = SummaryWriter(train_log_dir)
		val_writer = SummaryWriter(val_log_dir)

	opts.save_txt('config.txt', log_dir_num)

	### Load from Checkpoint
	if resume_from_epoch is not None:
		try:
			initial_epoch = model.load_checkpoint(resume_from_epoch) + 1
			iterations = (initial_epoch -1)*opts.batch_size
		except FileNotFoundError:
			print('No model available for epoch {}, starting fresh'.format(resume_from_epoch))
			initial_epoch = 1
			iterations = 0

	else:
		initial_epoch = 1
		iterations = 0

	### TRAIN AND VALIDATE ###
	if opts.jobs == 1:
		opts.best_model = 1e12

	if opts.hyperparameter == True:
		best_loss_this_run = None

	# MODEL PARAMETERS
	opt = torch.optim.Adam(model.parameters(), lr=opts.lr)

	if opts.loss_type == 'l1':
		loss_function = nn.L1Loss()
	elif opts.loss_type == 'mse':
		loss_function = nn.MSELoss()
	
	for epoch in range(initial_epoch, num_epochs + 1):
		epoch_start = time.perf_counter()

		# TRAIN
		epoch_train_loss = None
		set_mode('train', model)

		bar = progress.bar.Bar('Epoch {} train'.format(epoch), max=len(train_loader))

		for data in train_loader:
			image, image_data = set_data(data, opts)
			loss_p = optimize(model, opt, loss_function, opts, image, image_data)
			if opts.loss_type == 'mse':
				loss_p = torch.sqrt(loss_p)
			if epoch_train_loss is None:
				epoch_train_loss = get_errors(loss_p)
			else:
				epoch_train_loss = utils.concatenate_dicts(epoch_train_loss, get_errors(loss_p))
			
			gc.collect()
			iterations += 1
			bar.next()
		bar.finish()

		train_end = time.perf_counter()

		# VALIDATE
		epoch_val_loss = None
		set_mode('eval', model)

		bar = progress.bar.Bar('Epoch {} val'.format(epoch), max=len(val_loader))

		for data in val_loader:
			image, image_data = set_data(data, opts)
			loss_p, _ = model_test(opts, model, loss_function, image, image_data, compute_loss=True)
			if opts.loss_type == 'mse':
				loss_p = torch.sqrt(loss_p)
			if epoch_val_loss is None:
				epoch_val_loss = get_errors(loss_p)
			else:
				epoch_val_loss = utils.concatenate_dicts(epoch_val_loss, get_errors(loss_p))
			bar.next()
		bar.finish()

		epoch_end = time.perf_counter()

		epoch_avg_val_loss = utils.compute_dict_avg(epoch_val_loss)
		epoch_avg_train_loss = utils.compute_dict_avg(epoch_train_loss)
		train_fps = len(train_data)/(train_end-epoch_start)
		val_fps = len(val_data)/(epoch_end-train_end)

		print('End of epoch {}/{} | iter: {} | time: {:.3f} s | train: {:.3f} fps | val: {:.3f} fps'.format(epoch, num_epochs, iterations, epoch_end - epoch_start, train_fps, val_fps))

		# LOG ERRORS
		train_errors = utils.tag_dict_keys(epoch_avg_train_loss, 'train')
		val_errors = utils.tag_dict_keys(epoch_avg_val_loss, 'val')
		print('Train errors: ', train_errors)
		print('Val errors: ', val_errors)
		for key, value in sorted(train_errors.items()):
			# print('Key: ', key, 'Value: ', value)
			train_writer.add_scalar(key, value, epoch)
			print('{:20}: {:.3e}'.format(key, value))

		for key, value in sorted(val_errors.items()):
			# print('Key: ', key, 'Value: ', value)
			val_writer.add_scalar(key, value, epoch)
			print('{:20}: {:.3e}'.format(key, value))

		# SAVE CHECKPOINT
		save_checkpoint(epoch, 'latest', opts, model)

		if epoch % opts.checkpoint_interval == 0:
			save_checkpoint(epoch, epoch, opts, model)

		curr_total_val_loss = 0
		for key, val in epoch_avg_val_loss.items():
			try:
				curr_total_val_loss += val[-1]
			except IndexError:
				curr_total_val_loss += val
		
		if curr_total_val_loss < opts.best_model:
			save_checkpoint(epoch, 'best', opts, model)
			opts.best_model = curr_total_val_loss

			# save the config of the best performing model
			opts.save_txt('best_model_config.txt')
			print('\nThe current best model hyperparameters are: \n')
			print(opts)
		

		# Early stopping if validation loss is not decreasing sufficiently
		# if opts.hyperparameter == True:
		# 	if  best_loss_this_run == None:
		# 		best_loss_this_run = curr_total_val_loss
		# 	elif curr_total_val_loss < best_loss_this_run:
		# 		best_loss_this_run = curr_total_val_loss

		# 	if curr_total_val_loss > best_loss_this_run + 0.2:
		# 		print('\n*** The validation error is increasing. Training complete. ***')
		# 		break

def test(opts, model, test_data, which_epoch='best', batch_size=1, save_loss=False):
	test_loader = DataLoader(test_data,
		                     batch_size = batch_size,
		                     shuffle=False,
		                     num_workers=opts.dataloader_workers,
		                     pin_memory=True)

	model = load_checkpoint(opts, model, which_epoch)
	set_mode('eval', model)

	output_dir = os.path.join(opts.results_dir, opts.experiment_name, 'test_{}'.format(which_epoch))

	os.makedirs(output_dir, exist_ok=True)

	test_start = time.perf_counter()

	test_loss = None

	if opts.loss_type == 'l1':
		loss_function = nn.L1Loss()
	elif opts.loss_type == 'mse':
		loss_function = nn.MSELoss()

	bar = progress.bar.Bar('Test', max=len(test_loader))
	for idx, data in enumerate(test_loader):
		image, image_data = set_data(data, opts)
		_, delta_param = model_test(opts, model, loss_function, image, image_data, False)

		if save_loss:
			if test_loss is None:
				test_loss = get_data(opts, image, image_data, delta_param)
			else:
				test_loss = utils.concatenate_dicts(test_loss, get_data(opts, image, image_data, delta_param))

		bar.next()
	bar.finish()

	test_end = time.perf_counter()
	test_fps = len(test_data)/(test_end-test_start)
	print('Processed {} images | time: {:.3f} s | test: {:.3f} fps'.format(len(test_data), test_end-test_start, test_fps))

	if save_loss:
		loss_file = os.path.join(output_dir, 'loss.csv')
		header = [key for key in test_loss]
		entries = [test_loss[key] for key in test_loss]
		entries = np.atleast_2d(np.array(entries)).T.tolist()

		print("Saving test loss to {}".format(loss_file))
		with open(loss_file, 'wt') as file:
			file.write(','.join(header) + '\n')
			for entry in entries:
				line = ','.join([str(val) for val in entry]) + '\n'
				file.write(line)