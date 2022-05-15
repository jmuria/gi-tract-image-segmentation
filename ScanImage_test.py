import unittest
from  ScanImage import ScanImage
import matplotlib.pyplot as plt
import numpy as np

def apply_mask(image, maskImage):  
        image = image / image.max()
        image = np.dstack((image, maskImage,image)) 
        return image

def ShowImage(maskImage,title):
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    ax.set_title(title+' ('+str(maskImage.shape[0])+','+str(maskImage.shape[1])+')')
    ax.imshow(maskImage)    
    plt.show()

class TestScanImage(unittest.TestCase):
     
     def test_ICanOpenAnImage(self):
        filePath='..\\input\\uw-madison-gi-tract-image-segmentation\\train\\case101\\case101_day20\\scans\\slice_0001_266_266_1.50_1.50.png'
        image=ScanImage.Create(filePath)  
        self.assertEqual(image.shape[0], 266)
        self.assertEqual(image.shape[1], 266)

     def test_ICanResizeA266x266Into360x360WithoutScaling(self):
        filePath='..\\input\\uw-madison-gi-tract-image-segmentation\\train\\case101\\case101_day20\\scans\\slice_0001_266_266_1.50_1.50.png'
        image=ScanImage.Create(filePath) 
        image=ScanImage.ResizeWithoutScaling(image,360,360)
        self.assertEqual(image.shape[0], 360)        
        self.assertEqual(image.shape[1], 360)

     def test_ICanResizeA266x266Into200x200WithoutScaling(self):
        filePath='..\\input\\uw-madison-gi-tract-image-segmentation\\train\\case101\\case101_day20\\scans\\slice_0001_266_266_1.50_1.50.png'
        image=ScanImage.Create(filePath) 
        image=ScanImage.ResizeWithoutScaling(image,200,200)
        self.assertEqual(image.shape[0], 200)        
        self.assertEqual(image.shape[1], 200)
        ShowImage(image,"Reduced")
    
     def test_ICanConvertA1_63mmxPxIntoA1_50mmxPx(self):
        filePath='..\\input\\uw-madison-gi-tract-image-segmentation\\train\\case117\\case117_day13\\scans\\slice_0001_276_276_1.63_1.63.png'
        image=ScanImage.Create(filePath) 
        self.assertEqual(image.shape[0], 276)        
        self.assertEqual(image.shape[1], 276)
        image=ScanImage.ConvertPixelSize(image,1.63,1.50)
        ShowImage(image,"Original")
        self.assertEqual(image.shape[0], 300)        
        self.assertEqual(image.shape[1], 300)