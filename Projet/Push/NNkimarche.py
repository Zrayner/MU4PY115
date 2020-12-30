from __future__ import print_function
import keras,sklearn
# suppress tensorflow compilation warnings
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
seed=0
np.random.seed(seed) # fix random seed

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import pickle






def create_model():
    # instantiate model
    model = Sequential()
    # add a dense all-to-all tanh layer
    model.add(Dense(30, activation='tanh'))
    # add a dense all-to-all tanh layer
    model.add(Dense(30, activation='tanh'))
    # apply dropout with rate 0.5
  
    model.add(Dense(1, activation='softmax'))
    
    return model

model_H=create_model()
model_O=create_model()


inputO1 = tf.keras.layers.Input(shape=(63,))
inputO2 = tf.keras.layers.Input(shape=(63,))
inputH1 = tf.keras.layers.Input(shape=(63,))
inputH2 = tf.keras.layers.Input(shape=(63,))
inputH3 = tf.keras.layers.Input(shape=(63,))
inputH4 = tf.keras.layers.Input(shape=(63,))
inputH5 = tf.keras.layers.Input(shape=(63,))

EO1=model_O(inputO1)
EO2=model_O(inputO2)
EH1=model_H(inputH1)
EH2=model_H(inputH2)
EH3=model_H(inputH3)
EH4=model_H(inputH4)
EH5=model_H(inputH5)

added= tf.keras.layers.Add()([EO1, EO2,EH1,EH2,EH3,EH4,EH5])
model_TOT=tf.keras.models.Model(inputs=[inputO1, inputO2, inputH1 , inputH2 , inputH3 , inputH4 , inputH5], outputs=added)

def compile_model(model):
    # create the mode
    # compile the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

model_TOT=compile_model(model_TOT)   
    
model_TOT.summary()   
    