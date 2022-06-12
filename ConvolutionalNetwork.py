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
                 batch_size,
                 num_classes=1
                ):
        self.images = images
        self.masks= masks  
        self.batch_size=batch_size
        self.target_shape=target_shape
        self.num_classes=num_classes
        
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_input_img_paths = self.images[i : i + self.batch_size]
        
        x = np.zeros((self.batch_size,) + self.target_shape + (1,), dtype="float32")
        for j, img in enumerate(batch_input_img_paths):
            img = np.expand_dims(img, axis=-1)            
            x[j] = img
        
        if(np.any(self.masks!=None)):
            batch_target_img_paths = self.masks[i : i + self.batch_size]

            y = np.zeros((self.batch_size,) +  self.target_shape + (self.num_classes,), dtype="float32")
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

import keras.backend as K

#pip install focal-loss
#from focal_loss import BinaryFocalLoss
#loss=BinaryFocalLoss(gamma=2)
#segmentation-models
#CategoricalFocalLoss

def DiceLoss(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return 1-(2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
 
from focal_loss import BinaryFocalLoss

class ConvolutionalNetwork:
    model = Any

    def CreateModel(self):
        self.model = models.Sequential()
        return self.model
    
    def PrepareInput(self,width,height,nChannels):        
        self.Input= layers.Input(shape=(width,height,nChannels))
        #self.Input = layers.Lambda(lambda x: x / 255)( self.Input)
        #self.PreviousLayer = layers.Conv2D(32, 3, strides=2, padding="same")(self.Input)
        self.PreviousLayer = self.Input
    

    def PrepareIntermediateFilters(self):
            
        #Contraction path
        
        c1 = layers.Conv2D(16, (3, 3),activation='relu', kernel_initializer='he_normal', padding="same",name="c1")(self.PreviousLayer)
        c1 = layers.Dropout(0.1)(c1)        
        c1 = layers.Conv2D(16, (3, 3),activation='relu', kernel_initializer='he_normal', padding="same",name="c12")(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        
        c2 = layers.Conv2D(32, (3, 3),activation='relu', kernel_initializer='he_normal', padding="same",name="c2")(p1)
        c2 = layers.Dropout(0.1)(c2)        
        c2 = layers.Conv2D(32, (3, 3),activation='relu', kernel_initializer='he_normal', padding="same",name="c22")(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)
        
        c3 = layers.Conv2D(64, (3, 3),activation='relu', kernel_initializer='he_normal', padding="same",name="c3")(p2)
        c3 = layers.Dropout(0.2)(c3)        
        c3 = layers.Conv2D(64, (3, 3),activation='relu', kernel_initializer='he_normal', padding="same",name="c32")(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        
        c4 = layers.Conv2D(128, (3, 3),activation='relu', kernel_initializer='he_normal', padding="same",name="c4")(p3)
        c4 = layers.Dropout(0.2)(c4)        
        c4 = layers.Conv2D(128, (3, 3),activation='relu', kernel_initializer='he_normal', padding="same",name="c42")(c4)
        p4 = layers.MaxPooling2D((2, 2))(c4)
        

        c5 = layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding="same",name="cc5")(p4)
        c5 = layers.Dropout(0.3)(c5)
        c5 = layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding="same",name="c52")(c5)
        
    
        #Expansive path 
        u6 = layers.Conv2DTranspose(128, (2, 2),strides=(2, 2), padding="same",name="u6")(c5)
        u6 = layers.concatenate([u6, c4])        
        c6 = layers.Conv2D(128, (3, 3),activation='relu', kernel_initializer='he_normal', padding="same",name="c6")(u6)
        c6 = layers.Dropout(0.2)(c6)
        c6 = layers.Conv2D(128, (3, 3),activation='relu', kernel_initializer='he_normal', padding="same",name="c62")(c6)

        u7 = layers.Conv2DTranspose(64, (2, 2),strides=(2, 2), padding="same",name="u7")(c6)
        u7 = layers.concatenate([u7, c3])        
        c7 = layers.Conv2D(64, (3, 3),activation='relu', kernel_initializer='he_normal', padding="same",name="c7")(u7)
        c7 = layers.Dropout(0.2)(c7)
        c7 = layers.Conv2D(64, (3, 3),activation='relu', kernel_initializer='he_normal', padding="same",name="c72")(c7)

        u8 = layers.Conv2DTranspose(32, (2, 2),strides=(2, 2), padding="same",name="u8")(c7)
        u8 = layers.concatenate([u8, c2])        
        c8 = layers.Conv2D(32, (3, 3),activation='relu', kernel_initializer='he_normal', padding="same",name="c8")(u8)
        c8 = layers.Dropout(0.1)(c8)
        c8 = layers.Conv2D(32, (3, 3),activation='relu', kernel_initializer='he_normal', padding="same",name="c82")(c8)

        u9 = layers.Conv2DTranspose(16, (2, 2),strides=(2, 2), padding="same",name="u9")(c8)
        u9 = layers.concatenate([u9, c1], axis=3)        
        c9 = layers.Conv2D(16, (3, 3),activation='relu', kernel_initializer='he_normal', padding="same",name="c9")(u9)
        c9 = layers.Dropout(0.1)(c9)
        c9 = layers.Conv2D(16, (3, 3),activation='relu', kernel_initializer='he_normal', padding="same",name="c92")(c9)
        self.PreviousLayer=c9
    


     

    

    def PrepareOutput(self,width,height,nChannels):        
        self.Output = layers.Conv2D(nChannels, (1,1), activation="softmax")(self.PreviousLayer)
         

  
    def loss(self):
        #loss=DiceLoss
        #loss=BinaryFocalLoss(gamma=5)
        #loss="categorical_crossentropy"
        loss='binary_crossentropy'
        return loss

    
    
    def optimizer(self):
        from keras.optimizers import SGD
        #optimizer='adam'
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
        #optimizer = SGD(learning_rate=0.01)
        return optimizer
    
    def CompileModel(self):
        self.model = tf.keras.Model(self.Input, self.Output, name="U-Net")
        
        try:
            self.model.compile(
                optimizer=self.optimizer(),
                loss=self.loss(),                
                metrics=['accuracy'])
        except BaseException as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise
    
    def SaveModel(self,modelPath):
        self.model.save(modelPath)

    def LoadModel(self,modelPath):
        self.model=tf.keras.models.load_model(modelPath)
        return self.model

    def Train(self,X,Y,image_shape,batch_size=1,epochs=10,num_classes=4):
        

        data=OrganDataset(X,Y,image_shape,batch_size,num_classes=4)

        history=self.model.fit(data, epochs=epochs 
                    #,validation_data=(test_images, test_labels)
                     )
        return history


    def Predict(self,testImages,image_shape):
        
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
    