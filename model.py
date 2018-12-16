# Import Statements
import keras
import numpy as np
import sklearn.metrics as metrics
import keras.backend as K
from keras import layers
from keras import backend
from keras import optimizers
from keras.layers import LeakyReLU

# Define neural network stuff
learning_rate = 0.0048
clip_value = 1.0


# Define a custom optimizer that may be used
optimizer = optimizers.sgd(lr=learning_rate, clipvalue=clip_value)

# Peak Signal to Noise Ratio
def psnr(y_true,y_pred):
    mse = K.mean(y_true - y_pred) ** 2
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    result = 20 * K.log(PIXEL_MAX / K.sqrt(mse))
    return result

# CNN Model
def build_model():
    layer_one_input = keras.Input(shape=(240, 320, 3))
    lays = layers.Conv2D(96,(11,11),strides=4,activation='relu')(layer_one_input)
    lays = layers.AveragePooling2D(pool_size=2)(lays)
    lays = layers.Conv2D(256,(5,5),activation='relu')(lays)
    lays = layers.AveragePooling2D(pool_size=2)(lays)
    lays = layers.Conv2D(384,(3,3),activation='relu')(lays)
    lays = layers.Dense(256)(lays)
    lays = layers.Dense(128)(lays)
    lays = layers.Reshape(target_shape=(120,160))(lays)

    model = keras.models.Model(inputs=[layer_one_input], outputs=lays)
    model.compile(optimizer=optimizer, loss='mean_squared_logarithmic_error', metrics=['accuracy', psnr])
    return model

def build_paper_model():
    layer_one_input = keras.Input(shape=(240, 320, 3))
    lays = layers.Conv2D(96,(11,11),strides=4,activation='relu')(layer_one_input)
    lays = layers.AveragePooling2D(pool_size=2)(lays)
    lays = layers.Conv2D(256,(5,5),activation='relu')(lays)
    lays = layers.AveragePooling2D(pool_size=2)(lays)
    lays = layers.Conv2D(384,(3,3),activation='relu')(lays)
    lays = layers.AveragePooling2D()(lays)
    lays = layers.Conv2D(384,(3,5),activation='relu')(lays)
    lays = layers.AveragePooling2D()(lays)
    lays = layers.Dense(256)(lays)
    lays = layers.Dropout(0.2)(lays)
    lays = layers.Dense(4800)(lays)

    #lays2 = layers.Conv2D(63,(9,9),strides=2,activation='relu')(layer_one_input)
    #lays2 = layers.AveragePooling2D(pool_size=2)(lays2)
    #lays2 = layers.Concatenate()([lays,lays2])
    #lays2 = layers.Conv2D(64,(5,5),strides=1,activation='relu')(lays2)
    #lays2 = layers.Conv2D(64,(5,5),strides=1,activation='relu')(lays2)
    lays = layers.Reshape(target_shape=(60,80))(lays)

    model = keras.models.Model(inputs=[layer_one_input], outputs=lays)
    model.compile(optimizer=optimizer, loss='mean_squared_logarithmic_error', metrics=['accuracy', psnr])
    return model
