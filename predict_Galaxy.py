# -*- coding: utf-8 -*-
import numpy as np
import h5py
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

dict_galaxy={
    0:'Disk, Face-on, No Spiral',
    1:'Smooth, Completely round',
    2:'Smooth, in-between round',
    3:'Smooth, Cigar shaped',
    4:'Disk, Edge-on, Rounded Bulge',
    5:'Disk, Edge-on, Boxy Bulge',
    6:'Disk, Edge-on, No Bulge',
    7:'Disk, Face-on, Tight Spiral',
    8:'Disk, Face-on, Medium Spiral',
    9:'Disk, Face-on, Loose Spiral'
    }

train = './model/modelFloat.h5'
weights = './model/weights.h5'
galaxy = load_model(train)
#galaxy.load_weights(weights)

def predict(file):
  x = load_img(file, target_size=(69,69,3))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  prediction = galaxy.predict(x)[0]
  return prediction

with h5py.File('Galaxy10.h5', 'r') as f:
    images = np.array(f['images'])
    labels = np.array(f['ans'])


print(dict_galaxy[np.argmax(predict('modo1.jpg'))])
print(dict_galaxy[np.argmax(predict('barredSpiral.jpg'))])

indice = np.where(labels==5)[0][5]
#print(indice)
image = images[indice:indice+1]

plt.imshow(images[indice])
plt.axis('off')
plt.show()
#print(np.argmax(galaxy.predict(image)))
#print(labels[indice])

#pred = galaxy.predict(images)
#comp = [np.argmax(x) for x in pred]
#print(sum(labels==comp)/len(images))

"""
model tiene 0.85 y un pucho mas

modelFloat tienw 0.83...

En teoria ya estan prediccionedo but is the same shit

CORREGUIR IMAGENES DE GOOGLE *Creo que funciona pero estoy pendejo*

"""