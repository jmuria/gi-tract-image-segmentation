import numpy as np
from PIL import Image
import cv2
import numpy as np
from skimage.transform import resize
import tensorflow as tf

class ScanImage:
    def Create(path):
        return  np.array(Image.open(path))

    '''
    def Create(path,Width=-1,Height=-1,nChannels=1):
        """ Load an image with the correct shape using only TF
        
        Args:
            path (tf.string): Path to the image to be loaded
            resize_to (tuple, optional): Size to reshape image
        
        Returns:
            3 channel tf.Constant image ready for training/inference
        
        """
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_png(img_bytes, channels=nChannels, dtype=tf.uint16)
        
        #img = 255.*(img/tf.reduce_max(img))
        #if(Width>0):
         #   img = tf.image.resize(img, (tf.constant(Width), tf.constant(Height)))
        return img'''


    def ResizeWithoutScaling(image,newWidth,newHeight):        
        addedHeight = newHeight-image.shape[0]
        if(addedHeight>0):        
            image = np.concatenate((np.zeros((addedHeight,image.shape[1])),image),axis=0)    
            addedWidth=newWidth-image.shape[1]
            rightFrame=np.zeros((newHeight,addedWidth))
            image= np.hstack((image,rightFrame))
        else:            
            image= image[0:newWidth, 0:newHeight]
        
        return image



    def ConvertPixelSize(image,oldPixelSize,newPixelSize):
        newWidth=round(image.shape[0]*oldPixelSize/newPixelSize,0)
        newHeight=round(image.shape[0]*oldPixelSize/newPixelSize,0)
        
       
        image = resize(image, (newWidth, newHeight))        
        return image
