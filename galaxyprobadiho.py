# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 12:26:03 2021

@author: Velasquez
"""

import h5py
import os
import numpy as np
from tensorflow.keras import utils
from tensorflow.keras import  layers, models
from sklearn.model_selection import train_test_split


#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


#From astroNN documentation
with h5py.File('Galaxy10.h5', 'r') as f:
    images = np.array(f['images'])
    labels = np.array(f['ans'])


labels = utils.to_categorical(labels, 10)
labels = labels.astype(np.float32)
images = images.astype(np.float32)


"""
indice = np.where(labels==8)[0]
print(labels[indice[0]])

print(images[0].shape)
plt.imshow(images[indice[0]])
plt.axis('off')
plt.show()

A[:len(A)//2]

"""

#X_train, X_test, y_train, y_test =train_test_split(images,labels,random_state=49)

""" Prueba uno (No predice)""""""
model = models.Sequential()
model.add(layers.Conv2D(128, (3, 3),input_shape=images[0].shape))
model.add(layers.MaxPooling2D((3, 3)))
model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
"""


""" Prueba tres (Ojala sirva)"""

model=models.Sequential()
model.add(layers.Conv2D(1024, (3,3), activation='relu' ,input_shape = images[0].shape  )) #2^10
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(256, (3,3),activation='relu')) #2^8
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3),activation='relu')) #2^6
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
model.summary()

history = model.fit(images,labels, epochs=10)

"""
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_test, y_test))
"""

"""
pruebas = set(labels)
print(list(pruebas))
labels = labels.astype(np.float32)
images = images.astype(np.float32)
print(labels[0])
shape = images[0].shape
print(shape)
"""

"""Save model"""
dir_model = 'model/'
if not os.path.exists(dir_model):
  os.mkdir(dir_model)
model.save('model/modelFloat.h5')
model.save_weights('model/weightsFloat.h5')
