# Import Statements
import cv2
import keras
import numpy as np
import pandas as pd
import model as mycnn

# Read the training data into a csv
df = pd.read_csv('train.csv',header=None, names=['Images','Depths'], nrows=10)

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

# Build the model and load the weights from a file
model = mycnn.build_model()
model.load_weights('model_result.h5')

# Predict images
p = model.predict(np.asarray(np_list_images))
predict = np.array(p)

#print(predict.shape)
#print(np_list_images.shape)

#cv2.imshow('Real Images',np_list_images[1])
#cv2.waitKey(0)

#cv2.imshow('Real Images',np_list_depths[1])
#cv2.waitKey(0)

# Save off files and look at images
test = cv2.normalize(predict[0], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
cv2.imshow('Fake Images',test)
cv2.imwrite("pic_normalized.png",test)
cv2.imwrite("generated_pic.png",predict[1])
cv2.imwrite("Expected pic.png",np_list_depths[1])