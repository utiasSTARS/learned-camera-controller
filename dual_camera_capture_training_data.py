import PySpin
import asyncio
import numpy as np
import random
import collections
import cv2
from PIL import Image

import torch
from torchvision import transforms
import model
import argparse

import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import ipdb

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('savePath', type=str)
args = parser.parse_args()

# CONFIG
capture_name = 'data_00'

SAVE_DIRS = [args.savePath+capture_name+'/cam_1', args.savePath+capture_name+'/cam_2' ]

# Camera serial numbers
serial_1 = '20010165' # Primary camera
serial_2 = '20025295' # Secondary camera

cam_framerate = 30
first_exp = 1000
first_gain = 5
next_exp = 0
next_gain = 0

exp_upper_limit = 30000
exp_lower_limit = 75
gain_upper_limit = 30.0
gain_lower_limit = 0.0

perturb_index = 0

NUM_IMAGES = 5000
NUM_SAVERS = 12

def primary_cam_setup(cam):
	"""
	This function takes the primary camera and sets up the software input triggering along with the GPIO output trigger for the second camera.

	:param cam: Camera to set up
	:type cam: CameraPtr
	:return: True if successful, False otherwise
	:rtype: bool
	"""
	try:
		result = True
		nodemap = cam.GetNodeMap()

		# Configure the camera to allow for chunk data
		result &= configure_chunk_data(nodemap)

		# Setup the pixel format
		result &= pixel_format(1, cam, 'BGR8')

		# Set up the primary camera output GPIO signal
		print('\n\t*** CONFIGURING HARDWARE OUTPUT ***')
		cam.LineSelector.SetValue(PySpin.LineSelector_Line2)
		cam.V3_3Enable.SetValue(True)
		print('\t\tCamera 1 Hardware output set to Line 2...')

		result &= trigger_selector(1, cam, 'FrameStart')
		result &= trigger_overlap(1, cam, 'ReadOut')
		result &= configure_trigger(1, cam, 'software')

		print("\n\t*** CONFIGURING CAMERA ***")
		result &= acquisition_mode(1, cam)				   # Continuous acquisition
		result &= framerate(1, cam)						   # Set the framerate
		result &= auto_exposure_mode(1, cam, 'Continuous') # Autoexposure = On
		result &= auto_gain_mode(1, cam, 'Continuous')	   # Autogain = On
		print('\n')

	except PySpin.SpinnakerException as ex:
		print('Error: %s' % ex)
		result = False

	return result

def secondary_cam_setup(cam):
	"""
	This function takes the secondary camera and sets up the hardware triggering for the GPIO input signal.

	:param cam: Camera to set up
	:type cam: CameraPtr
	:return: True if successful, False otherwise
	:rtype: bool
	"""
	try:
		result = True
		nodemap = cam.GetNodeMap()

		# Configure the camera to allow for chunk data
		result &= configure_chunk_data(nodemap)

		# Set up the pixel format
		result &= pixel_format(2, cam, 'BGR8')

		# Set up the secondary camera hardware trigger
		result &= configure_trigger(2, cam, 'hardware')

		print("\n\t*** CONFIGURING CAMERA ***")
		result &= acquisition_mode(2, cam)		    # Continuous
		result &= auto_exposure_mode(2, cam, 'Off') # Autoexposure = Off
		result &= exposure_change(cam, first_exp)   # Set starting exposure
		result &= auto_gain_mode(2, cam, 'Off')     # Autogain = Off
		result &= gain_change(cam, first_gain)		# Set starting gain
		print('\n')

	except PySpin.SpinnakerException as ex:
		print('Error: %s' % ex)
		result = False

	return result

def configure_chunk_data(nodemap):
	try:
		result = True
		print('\t*** CONFIGURING CHUNK DATA ***')

		# Activate chunk mode
		chunk_mode_active = PySpin.CBooleanPtr(nodemap.GetNode('ChunkModeActive'))

		if PySpin.IsAvailable(chunk_mode_active) and PySpin.IsWritable(chunk_mode_active):
			chunk_mode_active.SetValue(True)

		print('\t\tChunk mode activated...')

		# Enable all types of chunk data
		chunk_selector = PySpin.CEnumerationPtr(nodemap.GetNode('ChunkSelector'))

		if not PySpin.IsAvailable(chunk_selector) or not PySpin.IsReadable(chunk_selector):
			print('Unable to retrieve chunk selector. Aborting...\n')
			return False

		# Retrieve entries
		entries = [PySpin.CEnumEntryPtr(chunk_selector_entry) for chunk_selector_entry in chunk_selector.GetEntries()]

		print('\t\tEnabling entries...')

		# Iterate through our list and select each entry node to enable
		for chunk_selector_entry in entries:
			# Go to next node if problem occurs
			if not PySpin.IsAvailable(chunk_selector_entry) or not PySpin.IsReadable(chunk_selector_entry):
				continue

			# if (chunk_selector_entry.GetSymbolic() == 'ExposureTime') or (chunk_selector_entry.GetSymbolic() == 'Gain'):

			chunk_selector.SetIntValue(chunk_selector_entry.GetValue())

			chunk_str = '\t {}:'.format(chunk_selector_entry.GetSymbolic())

			# Retrieve corresponding boolean
			chunk_enable = PySpin.CBooleanPtr(nodemap.GetNode('ChunkEnable'))

			# Enable the boolean, thus enabling the corresponding chunk data
			if not PySpin.IsAvailable(chunk_enable):
				print('\t{} not available'.format(chunk_str))
				result = False
			elif chunk_enable.GetValue() is True:
				print('\t{} enabled'.format(chunk_str))
			elif PySpin.IsWritable(chunk_enable):
				chunk_enable.SetValue(True)
				print('\t{} enabled'.format(chunk_str))
			else:
				print('\t{} not writable'.format(chunk_str))
				result = False

	except PySpin.SpinnakerException as ex:
		print('Error: %s' % ex)
		result = False

	return result

def configure_trigger(i, cam, trigger):
	"""
	This function configures the camera to use a trigger. First, trigger mode is ensured to be off in order to select the trigger source. Trigger mode is then enabled, which has the camera capture only a single image upon the execution of the chosen trigger.

	:param cam: Camera to configure trigger for.
	:type cam: CameraPtr
	:return: True if successful, False otherwise.
	:rtype: bool
	"""

	print('\n\t*** CONFIGURING TRIGGER ***')
	try:
		result = True

		# Ensure trigger mode off
		# The trigger must be disabled in order to configure whether the source
		# is software or hardware.
		if cam.TriggerMode.GetAccessMode() != PySpin.RW:
			print('Unable to disable trigger mode (node retrieval). Aborting...')
			return False

		cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)

		print('\t\tCamera {} Trigger mode disabled...'.format(i))

		# Select trigger source
		# The trigger source must be set to hardware or software while trigger
		# mode is off.
		if cam.TriggerSource.GetAccessMode() != PySpin.RW:
			print('Unable to get trigger source (node retrieval). Aborting...')
			return False

		if trigger == 'software':
			cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)
		elif trigger == 'hardware':
			cam.TriggerSource.SetValue(PySpin.TriggerSource_Line3)
			# result &= trigger_selector(cam, 'FrameStart')
			result &= trigger_overlap(2, cam, 'ReadOut')

		print('\t\tCamera {} trigger source set to {}...'.format(i, trigger))

		# Turn trigger mode on
		# Once the appropriate trigger source has been set, turn trigger mode
		# on in order to retrieve images using the trigger.
		cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
		print('\t\tCamera {} Trigger mode turned back on...'.format(i))

	except PySpin.SpinnakerException as ex:
		print('Error: %s' % ex)
		return False

	return result

def trigger_selector(i, cam, setting):
	"""
	 :param cam: Camera to configure trigger for.
	 :type cam: CameraPtr
	 :return: True if successful, False otherwise.
	 :rtype: bool
	"""

	print('\n\t*** CONFIGURING TRIGGER SELECTOR ***')
	try:
		result = True
		
		if cam.TriggerSelector.GetAccessMode() != PySpin.RW:
			print('Unable to access trigger selector (node retrieval). Aborting...')
			return False

		if setting == 'AcquisitionStart':
			cam.TriggerSelector.SetValue(PySpin.TriggerSelector_AcquisitionStart)
		elif setting == 'FrameStart':
			cam.TriggerSelector.SetValue(PySpin.TriggerSelector_FrameStart)

		print('\t\tCamera {} Trigger Selector set to {}...'.format(i,setting))

	except PySpin.SpinnakerException as ex:
		print('Error: %s' % ex)
		return False

	return result

def trigger_overlap(i, cam, setting):
	"""
	 :param cam: Camera to configure trigger for.
	 :type cam: CameraPtr
	 :return: True if successful, False otherwise.
	 :rtype: bool
	"""

	print('\n\t*** CONFIGURING TRIGGER OVERLAP ***')
	try:
		result = True
		
		if cam.TriggerOverlap.GetAccessMode() != PySpin.RW:
			print('Unable to access trigger overlap (node retrieval). Aborting...')
			return False

		if setting == 'ReadOut':
			cam.TriggerOverlap.SetValue(PySpin.TriggerOverlap_ReadOut)
		elif setting == 'Off':
			cam.TriggerOverlap.SetValue(PySpin.TriggerOverlap_Off)

		print('\t\tCamera {} Trigger overlap set to {}...\n'.format(i, setting))

	except PySpin.SpinnakerException as ex:
		print('Error: %s' % ex)
		return False

	return result

def pixel_format(i, cam, setting):
	try:
		result = True

		print('\n\t*** CONFIGURING PIXEL FORMAT ***')
		node_pixel_format = PySpin.CEnumerationPtr(cam.GetNodeMap().GetNode('PixelFormat'))

		if not PySpin.IsAvailable(node_pixel_format) or not PySpin.IsWritable(node_pixel_format):
			print('Unable to set camera {} Pixel Format to {} (enum retrieval). Aborting...'.format(i, setting))

		node_pixel_format_setting = node_pixel_format.GetEntryByName(setting)
		if not PySpin.IsAvailable(node_pixel_format_setting) or not PySpin.IsReadable(node_pixel_format_setting):
			print('Unable to set camera {} Pixel Format to {} (entry retrieval). Aborting...'.format(i, setting))

		pixel_format_setting = node_pixel_format_setting.GetValue()
		node_pixel_format.SetIntValue(pixel_format_setting)
		
		print('\t\tCamera {} Pixel format set to {}...'.format(i, setting))

	except PySpin.SpinnakerException as ex:
		print('Error: %s' % ex)
		return False

	return result

def grab_next_image_by_trigger(cam):
	"""
	This function acquires an image by executing the trigger node.

	:param cam: Camera to acquire images from.
	:type cam: CameraPtr
	:return: True if successful, False otherwise.
	:rtype: bool
	"""
	try:
		result = True
		# Use trigger to capture image
			
		# Execute software trigger
		if cam.TriggerSoftware.GetAccessMode() != PySpin.WO:
			print('Unable to execute trigger. Aborting...')
			return False

		cam.TriggerSoftware.Execute()

	except PySpin.SpinnakerException as ex:
		print('Error: %s' % ex)
		return False

	return result

def print_device_info(nodemap, cam_num):
	"""
	This function prints the device information of each camera from the transport layer.

	:param nodemap: Transport layer device nodemap
	:param cam_num: Camera number
	:type nodemap: INodeMap
	:type cam_num: int
	:returns: True if successful, False otherwise
	:rtype: bool
	"""

	print('Printing device information for camera {}'.format(cam_num))

	try:
		result = True
		node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

		if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
			features = node_device_information.GetFeatures()

			for feature in features:
				node_feature = PySpin.CValuePtr(feature)
				print('{}: {}'.format(node_feature.GetName(), node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))

			else:
				print('Device control information not available.')
			print()


	except PySpin.SpinnakerException as ex:
		print('Error: {}'.format(ex))
		return False

	return result

def acquisition_mode(i, cam):
	try:
		result = True

		# Set acquisition mode to continuous
		node_acquisition_mode = PySpin.CEnumerationPtr(cam.GetNodeMap().GetNode('AcquisitionMode'))
		if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
			print('Unable to set acquisiton mode to continuous (node retrieval; camera {}). Aborting... \n'.format(i))
			return False

		node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
		if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(node_acquisition_mode_continuous):
			print('Unable to set acquistion mode to continuous (node entry \'Continuous\' retrieval camera {}). Aborting... \n'.format(i))
			return False

		acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

		node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

		print('\t\tCamera {} Acquisiton mode set to: Continuous...'.format(i))

	except PySpin.SpinnakerException as ex:
		print('Error: {}'.format(ex))
		result = False

	return result

def framerate(i, cam):
	try:
		result = True

		# SET ACQUISITION FRAME RATE AUTO TO OFF
		node_framerate_enable = PySpin.CBooleanPtr(cam.GetNodeMap().GetNode('AcquisitionFrameRateEnable'))
		if not PySpin.IsAvailable(node_framerate_enable) or not PySpin.IsWritable(node_framerate_enable):
			print('Unable to set FrameRateEnable to Off (node retrieval; camera {}). Aborting... \n'.format(i))
			return False

		node_framerate_enable.SetValue(True)
		print('\t\tCamera {} Manual FrameRate turned on...'.format(i))

		# SET ACQUISITION FRAMERATE TO 30
		node_acquisition_framerate = PySpin.CFloatPtr(cam.GetNodeMap().GetNode('AcquisitionFrameRate'))
		if not PySpin.IsAvailable(node_acquisition_framerate) or not PySpin.IsWritable(node_acquisition_framerate):
			print('Unable to set FrameRate to {} Hz (node retrieval; camera {}). Aborting... \n'.format(cam_framerate, i))
			return False

		# Set framerate
		node_acquisition_framerate.SetValue(cam_framerate)

		print('\t\tCamera {} Acquisition framerate set to %d...'.format(i, cam_framerate))

	except PySpin.SpinnakerException as ex:
		print('Error: {}'.format(ex))
		result = False

	return result

def auto_exposure_mode(i, cam, setting):
	try:
		result = True

		# SET AUTO EXPOSURE TO "SETTING"
		node_exposure_auto = PySpin.CEnumerationPtr(cam.GetNodeMap().GetNode('ExposureAuto'))
		if not PySpin.IsAvailable(node_exposure_auto) or not PySpin.IsWritable(node_exposure_auto):
			print('Unable to set FrameRateAuto to {} (node retrieval; camera {}). Aborting... \n'.format(setting, i))
			return False

		# Retrieve entry node from enumeration node
		node_exposure_auto_setting = node_exposure_auto.GetEntryByName(setting)
		if not PySpin.IsAvailable(node_exposure_auto_setting) or not PySpin.IsReadable(node_exposure_auto_setting):
			print('Unable to set ExposureAuto mode to {} (node entry \'{}\' retrieval camera {}). Aborting... \n'.format(setting, setting, i))
			return False

		# Set the upper limit on the autoexposure to 30 ms
		if setting == 'Continuous':
			node_exposure_auto_limit = PySpin.CFloatPtr(cam.GetNodeMap().GetNode('AutoExposureExposureTimeUpperLimit'))
			if not PySpin.IsAvailable(node_exposure_auto_limit) or not PySpin.IsWritable(node_exposure_auto_limit):
				print('Unable to set AutoExposure Upper Limit (node retrieval; camera {}). Aborting... \n'.format(i))
				return False
			node_exposure_auto_limit.SetValue(exp_upper_limit)
			print('\t\tCamera {} Auto Exposure upper limit set to {} ms...'.format(i, exp_upper_limit/1000))


		# Retrieve integer value from entry node
		exposure_off = node_exposure_auto_setting.GetValue()

		# Set integer value from entry node as new value of enumeration node
		node_exposure_auto.SetIntValue(exposure_off)

		print('\t\tCamera {} Auto Exposure turned to {}...'.format(i, setting))

	except PySpin.SpinnakerException as ex:
		print('Error: {}'.format(ex))
		result = False

	return result

def auto_gain_mode(i, cam, setting):
	try:
		result = True

		# SET AUTO GAIN TO "SETTING"
		node_gain_auto = PySpin.CEnumerationPtr(cam.GetNodeMap().GetNode('GainAuto'))
		if not PySpin.IsAvailable(node_gain_auto) or not PySpin.IsWritable(node_gain_auto):
			print('Unable to set GainAuto to {}} (node retrieval; camera {}). Aborting... \n'.format(setting, i))
			return False

		# Retrieve entry node from enumeration node
		node_gain_auto_off = node_gain_auto.GetEntryByName(setting)
		if not PySpin.IsAvailable(node_gain_auto_off) or not PySpin.IsReadable(node_gain_auto_off):
			print('Unable to set GainAuto mode to {} (node entry \'{}\' retrieval camera {}). Aborting... \n'.format(setting, setting, i))
			return False

		# Set the upper limit on the autogain to 30 dB
		if setting == 'Continuous':
			node_gain_auto_limit = PySpin.CFloatPtr(cam.GetNodeMap().GetNode('AutoExposureGainUpperLimit'))
			if not PySpin.IsAvailable(node_gain_auto_limit) or not PySpin.IsWritable(node_gain_auto_limit):
				print('Unable to set AutoGain Upper Limit (node retrieval; camera {}). Aborting... \n'.format(i))
				return False
			node_gain_auto_limit.SetValue(gain_upper_limit)
			print('\t\tCamera {} Auto gain upper limit set to {} ms...'.format(i, gain_upper_limit))

		# Retrieve integer value from entry node
		gain_off = node_gain_auto_off.GetValue()

		# Set integer value from entry node as new value of enumeration node
		node_gain_auto.SetIntValue(gain_off)

		print('\t\tCamera {} Auto Gain turned to {}...'.format(i, setting))

	except PySpin.SpinnakerException as ex:
		print('Error: {}'.format(ex))
		result = False

	return result

def exposure_change(cam, exp):
	try:
		result = True

		if cam.ExposureTime.GetAccessMode() != PySpin.RW:
			print('Unable to set exposure/shutter time. Aborting...')
			return False

		# Set the exposure time
		cam.ExposureTime.SetValue(exp)

	except PySpin.SpinnakerException as ex:
		print('Error: %s' % ex)

	return result

def gain_change(cam, gain):
	try:
		result = True

		if cam.Gain.GetAccessMode() != PySpin.RW:
			print('Unable to set gain. Aborting...')
			return False
			
		# Set the gain
		cam.Gain.SetValue(gain)

	except PySpin.SpinnakerException as ex:
		print('Error: %s' % ex)

	return result

def perturb_values(current_exp, current_gain, intensity):
	global perturb_index

	if perturb_index < 2:
		next_exp = int(current_exp*(1+np.random.rand(1)/1.3))
	else:
		next_exp = int(current_exp*(1-np.random.rand(1)/1.3))
	
	if (perturb_index == 1) or (perturb_index == 2):
		next_g = float(current_gain*(1-np.random.rand(1)/2))
	else:
		if current_gain < 5:
			next_g = float(random.randint(5,10))
		else:
			next_g = float(current_gain*(1+np.random.rand(1)))

	perturb_index += 1
	if perturb_index == 4:
		perturb_index = 0

	if next_exp > exp_upper_limit:
		next_exp = exp_upper_limit
	elif next_exp < exp_lower_limit:
		next_exp = exp_lower_limit

	if next_g > gain_upper_limit:
		next_g = gain_upper_limit
	elif next_g < gain_lower_limit:
		next_g = gain_lower_limit
	
	# print("The next exposure is {} and the next gain is {}".format(next_exp, next_gain))

	return next_exp, next_g

def reset_camera(i, cam):
	"""
	This function resets a camera to its default settings
	"""
	try:
		result = True

		print('*** RESETTING CAMERA SETTINGS ***')

		if cam.GetUniqueID() == serial_1:		
			result &= auto_exposure_mode(i, cam, 'Continuous')
			result &= auto_gain_mode(i, cam, 'Continuous')

			cam.V3_3Enable.SetValue(False)
			cam.LineSelector.SetValue(PySpin.LineSelector_Line0)
			cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)

		if cam.GetUniqueID() == serial_2:		
			result &= auto_exposure_mode(i, cam, 'Continuous')
			result &= auto_gain_mode(i, cam, 'Continuous')
			cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
		print('\n')

	except PySpin.SpinnakerException as ex:
		print('Error: %s' % ex)

	return result

async def acquire_images(queue: asyncio.Queue, cam: PySpin.Camera):
	"""
	A coroutine that captures 'NUM_IMAGES' images from 'cam' and puts them along with the camera serial number as a tuple into the 'queue'.
	"""
	# Setup camera
	cam_id = cam.GetUniqueID()

	if cam_id == serial_1:
		cam_name = 'Camera 1'
	elif cam_id == serial_2:
		cam_name = 'Camera 2'

	prev_frame_ID = 0
	global next_exp
	global next_gain

	# Acquisition Loop
	t0 = time.time()
	for i in range(NUM_IMAGES):

		# Software trigger for camera 1
		if cam_id == serial_1:
			grab_next_image_by_trigger(cam)

		img = cam.GetNextImage()
		frame_ID = img.GetFrameID()

		if img.IsIncomplete():
			print('WARNING: {}: image incomplete', frame_ID, 'with status'.format(cam_name), PySpin.Image_GetImageStatusDescription(img.GetImageStatus()))
			prev_frame_ID = frame_ID
			continue
		if frame_ID != prev_frame_ID + 1:
			print('WARNING: {}: skipped frame:'.format(cam_name), frame_ID)
		prev_frame_ID = frame_ID

		# Get exposure and gain from autoexposure, apply to second camera
		chunk_data = img.GetChunkData()
		exposure = chunk_data.GetExposureTime()
		gain_value = chunk_data.GetGain()

		# Get the current auto-exposure and auto-gain values
		if cam_id == serial_1:
			mean_intensities = cv2.mean(img.GetNDArray())
			intensity = cv2.mean(mean_intensities[0:2])[0]
			next_exp, next_gain = perturb_values(exposure, gain_value, intensity)

		# Pass the auto-exposure and auto-gain values to the second camera
		if cam_id == serial_2:

			exposure_change(cam, next_exp)
			gain_change(cam, next_gain)

		gain_value = float("{:.2f}".format(gain_value))
		queue.put_nowait((img, cam_id, exposure, gain_value))
		print('[{}] Acquired image {} with exposure {} and gain {}'.format(cam_name, frame_ID, int(exposure), gain_value))
		if cam_name == 'Camera 2':
			print('\n')
		await asyncio.sleep(0) # Necessary for context switches

	t1 = time.time()

	# Clean up
	await queue.join() # Wait for all images to be saved before EndAcq
	cam.EndAcquisition()

	print('Camera {}: Image capture took {:.4f} seconds. This corresponds to a framerate of {:.2f} Hz compared to actual {} Hz'.format(cam_name, (t1-t0), (NUM_IMAGES/(t1-t0)), cam_framerate))

async def save_images(queue: asyncio.Queue, save_dirs: dict, ext='.jpg'):
	"""
	A coroutine that gets images from the 'queue' and saves them using the global Thread Pool Executor.
	"""
	while True:
		# Receive image
		image, cam_id, exposure, gain = await queue.get()

		# Create filename
		frame_id = image.GetFrameID()
		image_name = capture_name+'_'+str(frame_id).zfill(4)+'_exp-{}_gain-{}'.format(int(exposure), gain)+ext
		filename = os.path.join(save_dirs[cam_id], image_name)

		# Save the image using a pool of threads
		await loop.run_in_executor(tpe, save_image, image, filename)
		queue.task_done()

def save_image(image: PySpin.Image, filename: str):
	image.Save(filename)

async def main():
	"""
	Setting up the camera system

	:return: True if successful, False otherwise
	:rtype: bool
	"""
	result = True

	# Set up system object
	system = PySpin.System.GetInstance()
	version = system.GetLibraryVersion()
	print('\n***** Setting up Camera *****')
	print('\nSpinnaker library version: {}.{}.{}.{}'.format(version.major, version.minor, version.type, version.build))

	# Retrieve list of cameras
	cam_list = system.GetCameras()
	num_cameras = cam_list.GetSize()
	queue = asyncio.Queue()

	print('Number of cameras detected: {}'.format(num_cameras))

	# Create save directories
	for DIR in SAVE_DIRS:
		os.makedirs(DIR, exist_ok=True)

	# Exit if there are no cameras:
	if num_cameras == 0:
		cam_list.Clear()
		system.ReleaseInstance()
		print('No cameras detected!')
		input('Done! Press Enter to exit...')
		return False

	# Match the serial numbers to save locations
	assert num_cameras <= len(SAVE_DIRS), 'More cameras than save directories'
	camera_sns = [cam.GetUniqueID() for cam in cam_list]
	save_dir_per_cam = dict(zip(camera_sns, SAVE_DIRS))

	# Configure cameras
	print('Configuring all cameras...\n')
	cam_1 = cam_list.GetBySerial(camera_sns[0])
	cam_2 = cam_list.GetBySerial(camera_sns[1])

	# Print device information for the camera
	print('*** DEVICE INFORMATION ***\n')
	nodemap_tldevice_1 = cam_1.GetTLDeviceNodeMap()
	nodemap_tldevice_2 = cam_2.GetTLDeviceNodeMap()
	print_device_info(nodemap_tldevice_1, 1)
	print_device_info(nodemap_tldevice_2, 2)

	# Initialize the cameras
	cam_1.Init()
	cam_2.Init()

	# Setup the hardware triggers
	# Primary
	print('*** CONFIGURING CAMERA 1 ***')
	result &= primary_cam_setup(cam_1)

	# Secondary
	print('*** CONFIGURING CAMERA 2 ***')
	result &= secondary_cam_setup(cam_2)

	cam_2.BeginAcquisition()
	cam_1.BeginAcquisition()

	# Start the acquisition and save coroutines
	acquisition = [asyncio.gather(acquire_images(queue, cam)) for cam in cam_list]
	savers = [asyncio.gather(save_images(queue, save_dir_per_cam)) for _ in range(NUM_SAVERS)]

	await asyncio.gather(*acquisition)
	
	print('\nAcquisition Complete!\n')

	result &= reset_camera(1, cam_1)
	result &= reset_camera(2, cam_2)

	cam_1.DeInit()
	cam_2.DeInit()

	del cam_1
	del cam_2

	# cancel the now idle savers
	for c in savers:
		c.cancel()

	# Clean up environment and shut down cameras properly
	cam_list.Clear()
	system.ReleaseInstance()

	input('Done! Press Enter to exit...')
	return result

loop = asyncio.get_event_loop()
tpe = ThreadPoolExecutor(None)
loop.run_until_complete(main())