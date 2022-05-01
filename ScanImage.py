import numpy as np
from PIL import Image
import cv2
import numpy as np
from skimage.transform import resize

class ScanImage:
    def Create(path):
        return  np.array(Image.open(path))



    def ResizeWithoutScaling(image,newWidth,newHeight):
        addedHeight = newHeight-image.shape[0]
        image = np.concatenate((np.zeros((addedHeight,image.shape[1])),image),axis=0)    

        addedWidth=newWidth-image.shape[1]
        rightFrame=np.zeros((newHeight,addedWidth))
        image= np.hstack((image,rightFrame))
        
        return image



    def ConvertPixelSize(image,oldPixelSize,newPixelSize):
        newWidth=round(image.shape[0]*oldPixelSize/newPixelSize,0)
        newHeight=round(image.shape[0]*oldPixelSize/newPixelSize,0)
        
       
        image = resize(image, (newWidth, newHeight))        
        return image
