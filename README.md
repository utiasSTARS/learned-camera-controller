# Learned Camera Exposure and Gain Controller

## Dependencies
- numpy
- pytorch + torchvision (1.2)
- Pillow
- progress (for progress bars in train/val/test loops)
- tensorboard + tensorboardX (for visualization)
- OpenCV
- pickle
- PySpin (Required only for controlling FLIR BlackFly S cameras)
- asyncio (Required only for controlling FLIR BlackFly S cameras)

The camera controller code is intended to work with FLIR Blackfly S Machine Vision cameras through the Spinnaker PySpin API. The network must be trained with data captured using the same camera and lens configuration for which it is intended.

To generate training targets and train the network, a dataset is required consisting of image trajectories acquired simultaneously with two side-by-side cameras and downsampled to a 3x224x224 resolution (See the code used to acquire the dataset used in this work: `dual_camera_capture_training_data.py`.

The training data must be organized in the following manner for e.g., data_01:\
`PATH_TO_DATASET/data_01/cam_1/data_01_####_exp-XXXX_gainX.X.jpg`\
`PATH_TO_DATASET/data_01/cam_2/data_01_####_exp-XXXX_gainX.X.jpg`

## Generate Training Targets
1. `cd data_generation`
2. Run\
`ipython generate_feature_targets.py PATH_TO_DATASET`,\
`ipython generate_gridsearch_targets.py PATH_TO_DATASET` and\
`python generate_hybrid_targets.py`\
to generate each type of target. Note that hybrid targets can only be generated after the feature and gridsearch targets have been generated.

## Training the Network
1. Generate training targets.
2. Set the training options specific to your system (`num_workers`, `batch_size`, `validation_sets`, etc.) by modifying the values in `options.py`. 
3. Run `python run.py STAGE TARGET_METHOD DATASET_PATH` with the optional flags `--hyperparameter` and `--crossValidation`.\
 STAGE - training/testing\
 TARGET_METHOD - type of training target generation method used (features, gridsearch, hybrid)\
 DATASET_PATH - path to the acquired dataset
4. In another terminal run `tensorboard --port [port] --logdir [path]` to start the visualization server, where `[port]` should be replaced by a numeric value (e.g., 60006) and `[path]` should be replaced by your local results directory.
5. Tune in to `localhost:[port]` and watch the action.

## Using the Network to Acquire Images
1. Copy the trained network `.pth.tar` file from `results/TARGET_METHOD/checkpoints` to the main directory.
2. Update the filename in `dual_camera_capture_experiment.py`
3. Run `python dual_camera_capture_experiment.py`
