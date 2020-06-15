# 사진 이름 변경

# import os
# def changeN(path, cName):
#     i = 1
#     for filename in os.listdir(path):
#         os.rename(path+filename, path+str(cName)+str(i)+'.jpg')
#         i += 1
# changeN('C:/Users/YEOREUM/Desktop/notsmile/','notsmile')


#사진 resize와 labeling

import tensorflow as tf
import numpy as np

img = tf.keras.preprocessing.image.load_img('C:/Users/YEOREUM/Desktop/smileornot/smile1.jpg')
img = img.resize((64,64))
array = tf.keras.preprocessing.image.img_to_array(img)

List = np.array(array)
List = List[np.newaxis,:,:,:]

print(List.shape)

smile = np.array([1])
notsmile = np.array([0])
label = smile

mg = tf.keras.preprocessing.image.load_img('C:/Users/YEOREUM/Desktop/smileornot/notsmile1.jpg')
img = img.resize((64,64))
array = tf.keras.preprocessing.image.img_to_array(img)
array = array[np.newaxis,:,:,:]
List = np.concatenate((List,array))
label = np.vstack([label,notsmile])
print(label)



for i in range(1,181):
    string = 'C:/Users/YEOREUM/Desktop/smileornot/smile%d.jpg' % i
    img = tf.keras.preprocessing.image.load_img(string)
    img= img.resize((64,64))

    array = tf.keras.preprocessing.image.img_to_array(img)
    array = array[np.newaxis, :, :, :]

    List = np.concatenate((List, array))
    label = np.vstack([label,smile])

    string = 'C:/Users/YEOREUM/Desktop/smileornot/notsmile%d.jpg' % i
    img = tf.keras.preprocessing.image.load_img(string)
    img= img.resize((64,64))

    array = tf.keras.preprocessing.image.img_to_array(img)
    array = array[np.newaxis, :, :, :]

    List = np.concatenate((List, array))
    label = np.vstack([label,notsmile])


List = List.reshape(-1,64*64*3)
List = List.astype('float32')
#List/=225

np.savetxt('C:/Users/YEOREUM/Desktop/smileornotDataset.csv',List, delimiter = ',')
np.savetxt('C:/Users/YEOREUM/Desktop/smileornotLabel.csv',label, delimiter = ',')
List = List.reshape(-1,64,64,3)

