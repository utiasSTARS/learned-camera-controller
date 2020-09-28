import torch
import numpy as np
from PIL import Image
from torchvision import transforms

class ToTensor(object):
	""" Convert ndarrays in sample to Tensor"""

	def __init__(self, opts):
		self.sequence = opts.sequence

	def __call__(self, sample):
		tens = transforms.ToTensor()

		images, target = sample['image'], sample['target']
		img_t = images[0]
		img_t1 = images[1]
		img_t2 = images[2]

		return {'image': [tens(img_t), tens(img_t1), tens(img_t2)], 'target': torch.from_numpy(target)}

class Resize(object):
	""" Convert the image to have a specified size """

	def __init__(self, output_size, opts):
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size
		self.sequence = opts.sequence

	def __call__(self, sample):

		resize = transforms.Resize(self.output_size)

		images, target = sample['image'], sample['target']
		img_t = images[0]
		img_t1 = images[1]
		img_t2 = images[2]

		img_t = resize(img_t)
		img_t1 = resize(img_t1)
		img_t2 = resize(img_t2)

		return {'image': [img_t, img_t1, img_t2], 'target': target}

class HorizontalFlip(object):
	""" Randomly flip the images horizontally. Perform the same operation to all three images. """

	def __init__(self, opts):
		self.sequence = opts.sequence

	def __call__(self, sample):

		randomflip = transforms.RandomHorizontalFlip()
		flip = transforms.RandomHorizontalFlip(p=1.0)

		images, target = sample['image'], sample['target']
		img_t = images[0]
		img_t1 = images[1]
		img_t2 = images[2]

		# randomly flip first image, if flipped, flip the other two
		img_t_f = randomflip(img_t)
		if img_t_f != img_t:
			img_t1 = flip(img_t1)
			img_t2 = flip(img_t2)

		return {'image': [img_t_f, img_t1, img_t2], 'target': target}

class VerticalFlip(object):
	""" Randomly flip the images vertically. Perform the same operation to all three images. """

	def __init__(self, opts):
		self.sequence = opts.sequence

	def __call__(self, sample):

		randomflip = transforms.RandomVerticalFlip()
		flip = transforms.RandomVerticalFlip(p=1.0)

		images, target = sample['image'], sample['target']
		img_t = images[0]
		img_t1 = images[1]
		img_t2 = images[2]

		# randomly flip first image, if flipped, flip the other two
		img_t_f = randomflip(img_t)
		if img_t_f != img_t:
			img_t1 = flip(img_t1)
			img_t2 = flip(img_t2)

		return {'image': [img_t_f, img_t1, img_t2], 'target': target}

class NormalizeImage(object):
	""" Normalize all images after rescaling to be between 0-1  """

	def __init__(self, opts):
		self.sequence = opts.sequence

	def __call__(self, sample):

		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
										 std=[0.229, 0.224, 0.225])

		images, target = sample['image'], sample['target']
		img_t = images[0]
		img_t1 = images[1]
		img_t2 = images[2]
		img_t = normalize(img_t)
		img_t1 = normalize(img_t1)
		img_t2 = normalize(img_t2)

		return {'image': [img_t, img_t1, img_t2], 'target': target}