#%%
import tensorflow as tf
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from numpy import argmax
from PIL import Image

model = load_model('C:/Users/YEOREUM/Desktop/smileornot.h5')

test1 = plt.imread('C:/Users/YEOREUM/Desktop/rs2.jpg')

plt.imshow(test1)

test_num = plt.imread('C:/Users/YEOREUM/Desktop/rs2.jpg')
print('the answer is : ',model.predict_classes(test_num.reshape(1,64,64,3)))


# %%
