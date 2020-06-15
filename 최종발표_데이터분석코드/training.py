#%%
import tensorflow as tf
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tensorflow import keras


List = np.loadtxt('C:/Users/YEOREUM/Desktop/smileornotDataset.csv', delimiter=',')
label = np.loadtxt('C:/Users/YEOREUM/Desktop/smileornotLabel.csv', delimiter=',')

List = List.reshape(-1,64,64,3)


# plt.imshow(List[30])
# print('result : ',label[30])

# w_grid = 5
# len_grid = 5

# n_training = len(List)
# fig,axes = plt.subplots(len_grid,w_grid,figsize = (25,25))
# axes = axes.ravel()

# for i in np.arange(0,w_grid*len_grid):
#     index = np.random.randint(0,n_training)
#     axes[i].imshow(List[index])
#     axes[i].set_title(label[index],fontsize = 25)
#     axes[i].axis('off')

#List = List / 255


from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.models import load_model

traindata = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    brightness_range=[0.5,1.5],
    zoom_range=[0.8,1.1],
    rotation_range=15.,
    channel_shift_range=25,
    horizontal_flip=True
)
traindata = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
)

train_batch = traindata.flow(List,label,batch_size=16,shuffle=True)

cnn_model = Sequential()
cnn_model.add(Conv2D(64, 6, 6,input_shape = (64, 64, 3),activation = 'relu' ))
cnn_model.add(MaxPool2D(pool_size =(2, 2)))
cnn_model.add(Dropout(0.2))
cnn_model.add(Conv2D(64, 5, 5, activation = 'relu'))
cnn_model.add(MaxPool2D(pool_size =(2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(output_dim = 128, activation = 'relu'))
cnn_model.add(Dense(output_dim = 1, activation = 'sigmoid'))

cnn_model.compile(loss = 'binary_crossentropy',optimizer = Adam(lr = 0.001), metrics = ['accuracy'])
epochs = 50

history =cnn_model.fit(train_batch, batch_size = None, nb_epoch = epochs, verbose = 1)
cnn_model.save('C:/Users/YEOREUM/Desktop/smileornot.h5')

model = keras.models.load_model('C:/Users/YEOREUM/Desktop/smileornot.h5',compile = False)
export_path = 'C:/Users/YEOREUM/Desktop'
model.save(export_path,save_format="tf")

save_model = 'C:/Users/YEOREUM/Desktop'
converter = tf.lite.TFLiteConverter.from_saved_model(save_model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open('C:/Users/YEOREUM/Desktop/smileornot.tflite','wb').write(tflite_model)

# %%
