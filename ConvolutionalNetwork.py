from pyexpat import model
from typing import Any
import tensorflow as tf
import sys
import numpy as np
from keras import datasets, layers, models,utils
import matplotlib.pyplot as plt



from keras import datasets, layers, models


class OrganDataset(tf.keras.utils.Sequence):
    def __init__(self, 
                 images,                 
                 masks,                 
                 # data_dir = data_dir,
                 target_shape,
                 #image_trans = transforms.ToTensor(),                 
                ):
        self.images = images
        self.masks= masks  
        self.batch_size=1
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = datasets.cifar10.load_data()      
        self.target_shape = target_shape        
        #self.target_image_res = target_image_res
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        #x = np.zeros((self.batch_size,) + self.target_shape + (3,), dtype="float32")
        x = np.zeros((self.batch_size,) + self.target_shape , dtype="float32")
        for j, img in enumerate(self.images):            
            x[j] = img
        y = np.zeros((self.batch_size,) +  self.target_shape + (1,), dtype="uint8")
        for j, img in enumerate(self.masks): 
            y[j] = np.expand_dims(img[0], axis=2)           
            #y[j] = y[j]+img[0]
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:    
            #y[j] -= 1       
        return x, y



class ConvolutionalNetwork:
    model = Any

    def CreateModel(self):
        self.model = models.Sequential()
        return self.model
    
    def PrepareInput(self,width,height,nChannels):        
        '''self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, nChannels)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(10))'''
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
        
        #self.Output = layers.Conv2DTranspose(nChannels, 1, padding="same", activation = "sigmoid",input_shape=(360, 360, 1))
        #self.model.add(layers.Conv2D(32, (3, 3), activation='sigmoid'))
    
    def CompileModel(self):
        self.model = tf.keras.Model(self.Input, self.Output, name="U-Net")

        try:
            self.model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])

        except BaseException as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

            
    def Train(self,X,Y,image_shape):

        '''
        X = tf.convert_to_tensor(X, dtype=tf.int8)
        X2=[]
        X2.append(X)

        Y2=[]
        for i  in range(len(Y)):
            Y[i]=tf.convert_to_tensor(Y[i], dtype=tf.int8)
        Y2.append(Y)
        '''

        data=OrganDataset(X,Y,image_shape)

        history=self.model.fit(data, epochs=10 
                    #,validation_data=(test_images, test_labels)
                     )

    def PlotModel(self):
        tf.keras.utils.plot_model(self.model,to_file='../output/Model_Diagram.png',show_shapes=True, show_layer_names=True, expand_nested=True)

    