from pyexpat import model
from typing import Any
import tensorflow as tf
import sys

from keras import datasets, layers, models
import matplotlib.pyplot as plt

class ConvolutionalNetwork:
    model = Any

    def CreateModel(self):
        self.model = models.Sequential()
        return self.model
    
    def PrepareInput(self,width,height,nChannels):
        self.Input= layers.Input(shape=(width,height,nChannels))
        #self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, nChannels)))
    
    def PrepareOutput(self,width,height,nChannels):
        self.Output = layers.Conv2D(nChannels, 1, padding="same", activation = "softmax")(self.Input)
        #self.model.add(layers.Conv2D(32, (3, 3), activation='sigmoid'))
    
    def CompileModel(self):
        self.model = tf.keras.Model(self.Input, self.Output, name="U-Net")

        try:
            self.model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        except BaseException as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise


    def PlotModel(self):
        tf.keras.utils.plot_model(self.model,to_file='Model_Diagram.png',show_shapes=True, show_layer_names=True, expand_nested=True)

    