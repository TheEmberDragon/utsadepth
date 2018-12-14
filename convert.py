# Import Statements
import h5py
import keras
import numpy as np
import sklearn.metrics as metrics
from matplotlib import pyplot as plt
from keras import layers


with h5py.File('nyu_depth_v2_labeled.mat', 'r') as file:
    print(list(file.keys()))
    #accelData = list(file['accelData'])
    images = np.array(file['images'])[:500]
    depths_orig = np.array(file['depths'])[:500]

images = np.moveaxis(images,1,-1)
depths = np.reshape(depths_orig, (-1, 640, 480))
downsample_rate = 64
depths = depths.reshape(-1,80,60,downsample_rate).mean(axis=3)
#print(accelData[0])
print(images.shape)
print(depths.shape)

width, height, depth = images[0].shape

optimizer = keras.optimizers.RMSprop(lr=0.01, clipvalue=1.0)

def build_model():
    layer_one_input = keras.Input(shape=(width, height, depth))
    l_0 = layers.AveragePooling2D()(layer_one_input)
    l_1 = layers.Conv2D(96, (11,11))(l_0)
    l_2 = layers.MaxPool2D()(l_1)
    l_3 = layers.Conv2D(96, (5,5))(l_2)
    l_4 = layers.MaxPool2D()(l_3)
    l_5 = layers.Conv2D(96, (3,3))(l_4)
    p   = layers.MaxPool2D()(l_5)

    l_6 = layers.Conv2D(96, (3,3))(p)
    l_7 = layers.MaxPool2D()(l_6)

    l_8 = layers.Reshape(target_shape=(-1,))(l_7)

    l_9 = layers.Dense(256)(l_8)
    l_10 = layers.Dense(4800)(l_9)

    l_11 = layers.Reshape(target_shape=(80,60))(l_10)

    model = keras.models.Model(inputs=[layer_one_input], outputs=l_11)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    return model

epochs = 100
model = build_model()
model.summary()
#model.fit(images,depths,epochs=epochs)
#model.save('saved_model.h5')


model.load_weights('saved_model_bad.h5')

p = model.predict(images[:10,:,:,:])

truth = np.array(depths[:10,:,:])
predict = np.array(p)

plt.imshow(images[2,:,:,:],interpolation='nearest')
plt.show()

plt.imshow(depths_orig[2,:,:],interpolation='nearest')
plt.show()

plt.imshow(depths[2,:,:],interpolation='nearest')
plt.show()

plt.imshow(predict[2,:,:],interpolation='nearest')
plt.show()