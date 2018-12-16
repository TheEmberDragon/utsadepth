# Import Statements 
import cv2
import keras
import model as mycnn
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from keras import optimizers
from skimage.transform import rescale, resize, downscale_local_mean

# Define number of epochs
epochs = 1000

# Read the training data into a csv
df = pd.read_csv('train.csv',header=None, names=['Images','Depths'], nrows=1000)

# Create empty lists to be used
np_list_images = []
np_list_depths = []

# Loop through the images and add them to the numpy array
for index, row in df.iterrows():
    # print(index, row['Images'], row['Depths'])
    
    im = cv2.imread(row['Images'],1)
    # print(im.shape)
    #cv2.imshow('Real',im)
    #cv2.waitKey(0)

    np_list_images.append(cv2.resize(im, (0,0), fx=0.5, fy=0.5))
    #print(image_reduced.shape)
    
    #cv2.imshow('Real',np_list_images[index])
    #cv2.waitKey(0) 

    im = cv2.imread(row['Depths'],0)
    np_list_depths.append(cv2.resize(im, (0,0), fx=0.125, fy=0.125))
    #cv2.imshow('Depth',depth_reduced)
    #cv2.waitKey(0)  

# Build the model and print a summary
model = mycnn.build_paper_model()
model.summary()

# Train the model
model.load_weights('saved_model1.h5')
model.fit(np.asarray(np_list_images),np.asarray(np_list_depths),epochs=epochs)
model.save('saved_model1.h5')

# Save every epoch
keras.callbacks.ModelCheckpoint('saved_model_checkpoint.h5',monitor='mean_squared_logarithmic_error',verbose=0,save_best_only=False, save_weights_only=False, mode='auto')
