from pyexpat import model
from typing import Any
import tensorflow as tf
import sys
import numpy as np
from keras import datasets, layers, models,utils
import matplotlib.pyplot as plt



from keras import datasets, layers, models

from OrgansInSlicesData import OrgansInSlicesData


class OrganDataset(tf.keras.utils.Sequence):
    def __init__(self, 
                 images,                 
                 masks,                                 
                 target_shape,
                 batch_size
                ):
        self.images = images
        self.masks= masks  
        self.batch_size=batch_size
        self.target_shape=target_shape
        
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_input_img_paths = self.images[i : i + self.batch_size]
        
        x = np.zeros((self.batch_size,) + self.target_shape + (1,), dtype="float32")
        for j, img in enumerate(batch_input_img_paths):
            img = np.expand_dims(img, axis=-1)            
            x[j] = img
        
        if(self.masks!=None):
            batch_target_img_paths = self.masks[i : i + self.batch_size]

            y = np.zeros((self.batch_size,) +  self.target_shape + (1,), dtype="uint8")
            for j, img in enumerate(batch_target_img_paths):            
                y[j] = img
      
        return x, y

class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6, gama=2):
        super(DiceLoss, self).__init__()
        self.name = 'NDL'
        self.smooth = smooth
        self.gama = gama

    def call(self, y_true, y_pred):
        y_true, y_pred = tf.cast(
            y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
        nominator = 2 * \
            tf.reduce_sum(tf.multiply(y_pred, y_true)) + self.smooth
        denominator = tf.reduce_sum(
            y_pred ** self.gama) + tf.reduce_sum(y_true ** self.gama) + self.smooth
        result = 1 - tf.divide(nominator, denominator)
        return result

class ConvolutionalNetwork:
    model = Any

    def CreateModel(self):
        self.model = models.Sequential()
        return self.model
    
    def PrepareInput(self,width,height,nChannels):        
        self.Input= layers.Input(shape=(width,height,nChannels))
        self.PreviousLayer = layers.Conv2D(32, 3, strides=2, padding="same")(self.Input)
    
    def PrepareIntermediateFilters(self):
        x = self.PreviousLayer
        previous_block_activation = x  # Set aside residual
        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128, 256]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        ### [Second half of the network: upsampling inputs] ###

        for filters in [256, 128, 64, 32]:
            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters, 1, padding="same")(residual)
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        self.PreviousLayer=x

    def PrepareOutput(self,width,height,nChannels):        
        self.Output = layers.Conv2D(nChannels, 3, activation="sigmoid", padding="same")(self.PreviousLayer)
        

    
    def CompileModel(self):
        self.model = tf.keras.Model(self.Input, self.Output, name="U-Net")
        
        try:
            self.model.compile(
                optimizer='adam',
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),                
                #optimizer="rmsprop", 
                #loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])
        except BaseException as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

            
    def Train(self,X,Y,image_shape,batch_size=1,epochs=10):
        

        data=OrganDataset(X,Y,image_shape,batch_size)

        history=self.model.fit(data, epochs=epochs 
                    #,validation_data=(test_images, test_labels)
                     )
        return history


    def Predict(self,testImages,image_shape):
        #data=OrganDataset(testImages,None,image_shape,len(testImages))
        x = np.zeros((len(testImages),) + image_shape , dtype="float32")
        for j, img in enumerate(testImages):            
            x[j] = img
        val_preds = self.model.predict(x)
        return val_preds

    def PlotModel(self):
        tf.keras.utils.plot_model(self.model,to_file='../output/Model_Diagram.png',show_shapes=True, show_layer_names=True, expand_nested=True)

    def PlotHistory(history):
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        #plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()

        '''
        # summarize history for accuracy
        plt.plot(history.history['loss'])
        #plt.plot(history.history['val_accuracy'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()'''
    