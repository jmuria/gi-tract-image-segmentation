import tensorflow as tf
from ConvolutionalNetwork import ConvolutionalNetwork
import numpy as np
import cv2

class OrganDataset2(tf.keras.utils.Sequence):
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
        
        x = np.zeros((self.batch_size,) + self.target_shape + (3,), dtype="float32")
        for j, img in enumerate(batch_input_img_paths):
            #img = np.expand_dims(img, axis=-1)
            img_float32 = np.float32(img)
            img = cv2.cvtColor(img_float32, cv2.COLOR_GRAY2RGB)  
            img = 255.*(img/tf.reduce_max(img))    
            #img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)            
            x[j] = img
        
        if(np.any(self.masks!=None)):
            batch_target_img_paths = self.masks[i : i + self.batch_size]

            y = np.zeros((self.batch_size,) +  self.target_shape + (self.num_classes,), dtype="float32")
            for j, img in enumerate(batch_target_img_paths):            
                y[j] = img
      
        return x, y


class ConvNetwork2(ConvolutionalNetwork):

    IMAGE_SHAPE = (224,224)
    nInputChannels=1
    classes=3

    def convolution_block(self,block_input, num_filters=256, kernel_size=3,
                        dilation_rate=1, padding="same", use_bias=False,):
        """ TBD """
        x = tf.keras.layers.Conv2D(filters=num_filters, 
                                kernel_size=kernel_size, 
                                dilation_rate=dilation_rate, 
                                padding=padding, 
                                use_bias=use_bias, 
                                kernel_initializer=tf.keras.initializers.HeNormal())(block_input)
        x = tf.keras.layers.BatchNormalization()(x)
        return tf.keras.layers.Activation("relu")(x)


    def DilatedSpatialPyramidPooling(self,dspp_input):
        """ TBD """
        x = tf.keras.layers.AveragePooling2D(pool_size=(self.HIGH_FEAT_LAYER_OUTPUT_SHAPE[-3], 
                                                        self.HIGH_FEAT_LAYER_OUTPUT_SHAPE[-2]))(dspp_input)
        x = self.convolution_block(x, kernel_size=1, use_bias=True)
        
        # Get layers to concatenate
        out_pool = tf.keras.layers.UpSampling2D(size=(self.HIGH_FEAT_LAYER_OUTPUT_SHAPE[-3]//x.shape[1], 
                                                    self.HIGH_FEAT_LAYER_OUTPUT_SHAPE[-2]//x.shape[2]), 
                                                interpolation="bilinear")(x)
        _out_layers = [out_pool,]+\
                    [self.convolution_block(dspp_input, 256, _k, _d) for _k, _d in zip((1,3,3,3), (1,6,12,18))]
        
        output = self.convolution_block(tf.keras.layers.Concatenate(axis=-1)(_out_layers), kernel_size=1)

        return output


    def DeeplabV3Plus(self,backbone, low_feat_layer, high_feat_layer, n_classes, weights="imagenet", dropout=0.2):
        
      
        encoder_bb = backbone(weights=weights, include_top=False, input_tensor=self.Input)
        
        x = encoder_bb.get_layer(high_feat_layer).output
        x = tf.keras.layers.Dropout(dropout)(x)
        x = self.DilatedSpatialPyramidPooling(x)    
        
        input_a = tf.keras.layers.UpSampling2D(size=(self.IMAGE_SHAPE[0]//4//x.shape[1], 
                                                    self.IMAGE_SHAPE[1]//4//x.shape[2]), 
                                            interpolation="bilinear")(x)
        input_b = encoder_bb.get_layer(low_feat_layer).output
        input_b = tf.keras.layers.Dropout(dropout)(input_b)
        input_b = self.convolution_block(input_b, num_filters=48, kernel_size=1)

        x = tf.keras.layers.Concatenate(axis=-1)([input_a, input_b])
        x = self.convolution_block(x)
        x = self.convolution_block(x)
        x = tf.keras.layers.UpSampling2D(size=(self.IMAGE_SHAPE[0]//x.shape[1], 
                                            self.IMAGE_SHAPE[1]//x.shape[2]), 
                                        interpolation="bilinear",)(x)
        x = tf.keras.layers.Dropout(dropout/2)(x)
        return x
        

    def PrepareInput(self,width,height,nChannels):  
        self.IMAGE_SHAPE = (width,height)      
        self.Input = tf.keras.layers.Input(shape=(*self.IMAGE_SHAPE, nChannels))
        self.PreviousLayer=self.Input
    
    def PrepareOutput(self,width,height,nChannels):        
        self.classes = nChannels
        self.Output =  tf.keras.layers.Conv2D(nChannels, kernel_size=(1, 1), padding="same")(self.PreviousLayer)

    def CompileModel(self):
        
        self.model=tf.keras.Model(inputs=self.Input, outputs=self.Output)
        try:
            self.model.compile(
                optimizer=self.optimizer(),
                loss=self.loss(),                
                metrics=['accuracy'])
        except BaseException as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise
    
    def PrepareIntermediateFilters(self):
        # If you change the backbone you will need to adjust this accordingly
        self.BACKBONE = tf.keras.applications.ResNet50
        self.RES_HIGH_FEAT_LAYER = "conv4_block6_2_relu"
        self.RES_LOW_FEAT_LAYER = "conv2_block3_2_relu"
        _dummy_model = self.BACKBONE(include_top=False, weights=None, input_shape=(*self.IMAGE_SHAPE, self.nInputChannels))
        self.HIGH_FEAT_LAYER_OUTPUT_SHAPE = _dummy_model.get_layer(self.RES_HIGH_FEAT_LAYER).output_shape[1:]
        self.LOW_FEAT_LAYER_OUTPUT_SHAPE = _dummy_model.get_layer(self.RES_LOW_FEAT_LAYER).output_shape[1:]

        #Not used SUB_NODEBUG_MODEL_WT_PATH = "/kaggle/input/uwmgit-deeplabv3-end-to-end-pipeline-tf/resnet50_224x224x3_multiclass"

        
        

        MODEL_INSPECT = "summary"

        # We need this locally if we want to do all of this stuff without internet...
        #self.WEIGHT_PATH = "/kaggle/input/tf-keras-pretrained-model-weights/No Top/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
        self.WEIGHT_PATH = "../input/tf-keras-pretrained-model-weights/No Top/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"

        self.PreviousLayer=self.DeeplabV3Plus(backbone=self.BACKBONE, weights=self.WEIGHT_PATH,
                                        low_feat_layer=self.RES_LOW_FEAT_LAYER, 
                                        high_feat_layer=self.RES_HIGH_FEAT_LAYER, 
                                        n_classes=self.classes)
    
    def PrepareData(self,X,Y,image_shape,batch_size,num_classes):
        return OrganDataset2(X,Y,image_shape,batch_size,num_classes=num_classes)
    '''
    def Train(self,X,Y,image_shape,batch_size=1,epochs=10,num_classes=3):
        

        data=OrganDataset2(X,Y,image_shape,batch_size,num_classes=num_classes)

        history=self.model.fit(data,epochs=epochs,callbacks=self.PrepareCallbacks()
                     )
        return history
    '''        

    def Predict(self,testImages,image_shape):
               
        x = np.zeros((len(testImages),) + image_shape + (3,) , dtype="float32")
        for j, img in enumerate(testImages):    
            img_float32 = np.float32(img)            
            img = cv2.cvtColor(img_float32, cv2.COLOR_GRAY2RGB)    
            img = 255.*(img/tf.reduce_max(img))
            x[j] = img
        val_preds = self.model.predict(x)
        return val_preds