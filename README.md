# utsadepth
Repository to house a CNN for depth map generation. 

# Prerequisites
Pandas, Numpy, Tensorflow, Keras, SKLearn, and OpenCV

# Dataset
Dataset can be found at : https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html

# How to Run 
python dataprep.py -> Downloads NYU dataset and preps the data for training and testing.

python train.py -> Trains the CNN

python test.py -> Tests the CNN