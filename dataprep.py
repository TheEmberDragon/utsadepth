# Import Statements
import numpy as np
import h5py
from PIL import Image
import random
import wget
import os

# Function to conver mat files into python csv data file
# Code based off of: https://github.com/MasazI/cnn_depth_tensorflow/blob/master/prepare_data.py
def convert_data(path):
    img_dir = os.path.join("data","nyu_datasets")
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # Get the NYU Labeled data set
    data_url = 'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'

    # Download the data if it does not exist
    file = os.path.join("data","nyu_depth_v2_labeled.mat")
    if not os.path.exists(file):
        wget.download(data_url,out="data")

    # Load the data file
    f = h5py.File(path)

    # Create a training list
    train = []

    # Loop through the images in the data set
    for i, (image,depth) in enumerate(zip(f['images'],f['depths'])):
        
        # Get the raw image and depth from the dataset
        raw_image = image.transpose(2,1,0)
        raw_depth = depth.transpose(1,0)
        
        # Get the scaled depth 
        re_depth = (raw_depth/np.max(raw_depth))*255
        
        # Get a visual of the image and depth from the array
        image_pil = Image.fromarray(np.uint8(raw_image))
        depth_pil = Image.fromarray(np.uint8(re_depth))
        
        # Save the visual image as a jpg
        image_name = os.path.join("data","nyu_datasets", "%05d.jpg" % (i))
        image_pil.save(image_name)
        
        # Save the visual depth as a png
        depth_name = os.path.join("data","nyu_datasets", "%05d.png" % (i))
        depth_pil.save(depth_name)

        # Add the images to the training list
        train.append((image_name,depth_name))

    # Shuffle the training list
    random.shuffle(train)

    # Remove the training file if it is already there
  #  if not os.path.exists('train.csv'):
  #      os.remove('train.csv')

    # Write the image paths to a csv to use for training
    with open('train.csv', 'w+') as output:
        for(image_name,depth_name) in train:
            output.write("%s,%s" % (image_name, depth_name))
            output.write("\n")

# Main method
if __name__ == '__main__':
    # Get the current working directory
    current_dir = os.getcwd()
    data_path = 'data/nyu_depth_v2_labeled.mat'
    convert_data(data_path)