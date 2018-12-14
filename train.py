# Import Statements 
import cv2
import keras
import model as mycnn
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from keras import optimizers
from skimage.transform import rescale, resize, downscale_local_mean

# Define neural network stuff
learning_rate = 0.001
clip_value = 1.0
epochs = 300

# Define a custom optimizer that may be used
# optimizer = optimizers.sgd(lr=learning_rate, clipvalue=clip_value)

# Read the training data into a csv
df = pd.read_csv('train.csv',header=None, names=['Images','Depths'], nrows=1000)

# Create empty lists to be used
np_list_images = []
np_list_depths = []

# Loop through the images and add them to the numpy array
for index, row in df.iterrows():
    print(index, row['Images'], row['Depths'])
    
    im = cv2.imread(row['Images'],1)
    print(im.shape)
    #cv2.imshow('Real',im)
    #cv2.waitKey(0)

    np_list_images.append(cv2.resize(im, (0,0), fx=0.5, fy=0.5))
    #print(image_reduced.shape)
    
    #cv2.imshow('Real',np_list_images[index])
    #cv2.waitKey(0) 

    im = cv2.imread(row['Depths'],0)
    np_list_depths.append(cv2.resize(im, (0,0), fx=0.25, fy=0.25))
    #cv2.imshow('Depth',depth_reduced)
    #cv2.waitKey(0)  

# Build the model and print a summary
model = mycnn.build_model()
model.summary()

# Train the model
model.fit(np.asarray(np_list_images),np.asarray(np_list_depths),epochs=epochs)

# Save every epoch
keras.callbacks.ModelCheckpoint('saved_model.h5',monitor=psnr,verbose=0,save_best_only=False, save_weights_only=False, mode='auto')