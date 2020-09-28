# This script contains the code for generating feature-based labels for network training
import os
import time

import numpy as np
import cv2
import pickle
import concurrent.futures
import argparse
import ipdb

orb = cv2.ORB_create(nfeatures=10000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
print('OpenCV ORB matcher initialized...')

def max_features(windowed_images):
	max_num_features_orb = 0

	for im in windowed_images:
		img = cv2.imread(im,0)
		img = cv2.resize(img, (512, 384))

		# Get number of features in image
		kp = orb.detect(img, None)

		# Check if this is the highest number of features
		if len(kp) >= max_num_features_orb:
			best_image_orb = im
			max_num_features_orb = len(kp)

	print('The optimal number of orb features is {} and is from image: \n{}'.format(max_num_features_orb, best_image_orb))

	return best_image_orb

def process_dir(directory):
	cam_list = next(os.walk(directory))[1]

	trajectory = os.path.split(directory)[-1]

	print('*** Generating labels for {} ***\n'.format(trajectory))

	img_dir1 = os.path.join(directory, cam_list[0])
	img_dir2 = os.path.join(directory, cam_list[1])
	
	cam_1_images = next(os.walk(img_dir1))[2]
	cam_2_images = next(os.walk(img_dir2))[2]
	cam_1_images.sort()
	cam_2_images.sort()

	cam_1_paths = []
	cam_2_paths = []

	for im in cam_1_images:
		path = os.path.join(img_dir1, im)
		cam_1_paths.append(path)
	for im in cam_2_images:
		path = os.path.join(img_dir2, im)
		cam_2_paths.append(path)

	best_image_orb = []

	print('\nCurrently finding features in the trajectory: %s.\n' % trajectory)

	t0_folder = time.time()
	for i in range(len(cam_1_paths)-4):
		print('Finding maxmimum features in image {} at index {}'.format(i+3, i))
		if i == (len(cam_1_paths)-5):
			windowed_images = [cam_1_paths[i+2], cam_2_paths[i+2], cam_1_paths[i+3], cam_2_paths[i+3], cam_1_paths[i+4], cam_2_paths[i+4]]
		else:
			windowed_images = [cam_1_paths[i+2], cam_2_paths[i+2], cam_1_paths[i+3], cam_2_paths[i+3], cam_1_paths[i+4], cam_2_paths[i+4], cam_1_paths[i+5], cam_2_paths[i+5]]
		orb_features_output = max_features(windowed_images)
		best_image_orb.append(orb_features_output.split('/')[-1])
	t1_folder = time.time()
	m,s = divmod(t1_folder-t0_folder, 60)
	h,m = divmod(m, 60)
	print('Matching for {} is complete. This took {}:{}:{} (hours, min, secs)'.format(trajectory, int(h), int(m), int(s)))

	return best_image_orb

def main():
	### COMMAND LINE ARGUEMNTS ###
	parser = argparse.ArgumentParser()
	parser.add_argument('dataPath', type=str)
	args = parser.parse_args()

	# Create the dictionaries of the poses and images at each pose
	root_dir = args.dataPath
	
	root_dirs = []
	for folder in next(os.walk(root_dir))[1]:
		root_dirs.append(os.path.join(root_dir, folder))
	
	root_dirs.sort()
	root_dirs = root_dirs[40:41]

	with concurrent.futures.ProcessPoolExecutor() as executor:
		for directory, best_image_orb in zip(root_dirs, executor.map(process_dir, root_dirs)):
			trajectory = os.path.split(directory)[-1]

			os.makedirs('../data/features/', exist_ok=True)
			output_file_name_orb = '../data/features/optimal_images_orb_'+trajectory+'.pckl'
					
			f_optimal_orb = open(output_file_name_orb, 'wb')
			pickle.dump(best_image_orb, f_optimal_orb)
			f_optimal_orb.close()
	
if __name__ == '__main__':
	main()