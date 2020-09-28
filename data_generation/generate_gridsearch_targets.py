
import numpy as np
import cv2

import os
import time
import glob
import pickle
import concurrent.futures
import argparse

import ipdb

orb = cv2.ORB_create(nfeatures=2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
print('OpenCV ORB matcher initialized...')

def max_matches(currentImages, nextImages):
	"""
	Match images from frames t+1 and t, from both cameras. Find the settings that result in the most inlier feature matches between two frames. Save the settings of the second image as the optimal. for this version, perform this search over a window of 4 frames.
	"""
	max_num_inliers_orb = 0

	num_iters = int(len(currentImages)/2)	

	for i in range(num_iters):

		current_frame = currentImages[i*2:(i*2)+2]
		next_frame = nextImages[i*2:(i*2)+2]
		
		for curr_im in current_frame:
			img1 = cv2.imread(curr_im, 0)
			img1 = cv2.resize(img1, (512, 384))
			kp1, des1 = orb.detectAndCompute(img1, None)
			
			for j, next_im in enumerate(next_frame):

				img2 = cv2.imread(next_im,0)
				img2 = cv2.resize(img2, (512, 384))
				kp2, des2 = orb.detectAndCompute(img2, None)
				if des1 is not None and des2 is not None:
					orb_matches = bf.match(des1, des2)
					kp1_coords = np.asarray([kp1[m.queryIdx].pt for m in orb_matches]).reshape(-1,1,2)
					kp2_coords = np.asarray([kp2[m.trainIdx].pt for m in orb_matches]).reshape(-1,1,2)
					_, mask = cv2.findFundamentalMat(kp1_coords, kp2_coords, method=cv2.FM_RANSAC, ransacReprojThreshold=3.0)
					orb_inlier_matches = np.sum(mask)

					if orb_inlier_matches > max_num_inliers_orb:
						max_num_inliers_orb = orb_inlier_matches
						best_image_orb = next_im
						if j == 0:
							cam_orb = 1
						else:
							cam_orb = 2

	print('The optimal number of orb inlier matches is {} from camera {}'.format(max_num_inliers_orb, cam_orb))

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
		print('Finding maxmimum inlier matches for image {} at index {}'.format(i+3, i))
		if i == (len(cam_1_paths)-5):
			currentImages = [cam_1_paths[i+1], cam_2_paths[i+1], cam_1_paths[i+2], cam_2_paths[i+2], cam_1_paths[i+3], cam_2_paths[i+3]]
			nextImages = [cam_1_paths[i+2], cam_2_paths[i+2], cam_1_paths[i+3], cam_2_paths[i+3], cam_1_paths[i+4], cam_2_paths[i+4]]
		else:
			currentImages = [cam_1_paths[i+1], cam_2_paths[i+1], cam_1_paths[i+2], cam_2_paths[i+2], cam_1_paths[i+3], cam_2_paths[i+3], cam_1_paths[i+4], cam_2_paths[i+4]]
			nextImages = [cam_1_paths[i+2], cam_2_paths[i+2], cam_1_paths[i+3], cam_2_paths[i+3], cam_1_paths[i+4], cam_2_paths[i+4], cam_1_paths[i+5], cam_2_paths[i+5]]

		best_orb_output = max_matches(currentImages, nextImages)
		best_image_orb.append(best_orb_output.split('/')[-1])
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

	t0 = time.time()
	with concurrent.futures.ProcessPoolExecutor() as executor:
		for directory, best_image in zip(root_dirs, executor.map(process_dir, root_dirs)):
			trajectory = os.path.split(directory)[-1]

			os.makedirs('../data/gridsearch/', exist_ok=True)
			output_file_name_orb = '../data/gridsearch/optimal_images_orb_'+trajectory+'.pckl'
			
			f_optimal_orb = open(output_file_name_orb, 'wb')
			pickle.dump(best_image, f_optimal_orb)
			f_optimal_orb.close()

	t1 = time.time()
	
	m,s = divmod(t1-t0, 60)
	h,m = divmod(m,60)
	d,h = divmod(h,24)
	print('Completed matching! Everything took: {}:{}:{}:{} (days, hours, min, secs)'.format(int(d), int(h), int(m), int(s)))
	
if __name__ == '__main__':
	main()