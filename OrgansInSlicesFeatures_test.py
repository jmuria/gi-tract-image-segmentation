import unittest
from  OrgansInSlicesData import OrgansInSlicesData
import matplotlib.pyplot as plt
import numpy as np
from OrgansInSlicesFeatures import OrgansInSlicesFeatures

def ShowMask(maskImage,title):
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    ax.set_title(title+' ('+str(maskImage.shape[0])+','+str(maskImage.shape[1])+')')
    ax.imshow(maskImage)    
    plt.show()
 
class TestOrgansInSlicesFeatures(unittest.TestCase):

    basePath='..\\input\\uw-madison-gi-tract-image-segmentation\\train\\'
    databasePath='../input/uw-madison-gi-tract-image-segmentation/train.csv'


    def test_ICanGetTheFullPathOfASampleFromData(self):
        features=OrgansInSlicesFeatures(self.basePath)
        filePath='..\\input\\uw-madison-gi-tract-image-segmentation\\train\\case117\\case117_day13\\scans\\slice_0001_276_276_1.63_1.63.png'
        createdPath=features.CreatePath(117,13,1)
        self.assertEqual(createdPath, filePath)

    def test_ICanGetTheImageAndPixelSizesFromAPath(self):
        
        filePath='..\\input\\uw-madison-gi-tract-image-segmentation\\train\\case117\\case117_day13\\scans\\slice_0001_276_276_1.63_1.63.png'
        width,height,pixelSize=OrgansInSlicesFeatures.GetSizesFromPath(filePath)
        self.assertEqual(width,276)
        self.assertEqual(height,276)
        self.assertEqual(pixelSize,1.63)

    def test_ICanPrepareOneCase(self):
        features= OrgansInSlicesFeatures(self.basePath)
        x,y=features.Prepare(self.databasePath,1,368,368,1.50)
        self.assertEqual(len(x), 1)
        self.assertEqual(len(y), 1)
        self.assertEqual(x[0].shape, (368,368))
        self.assertEqual(y[0].shape, (368,368,1))
        ShowMask(x[0],"Image 0")
        ShowMask(y[0],"Mask 0") 


    def test_ICanPrepareTwoCases(self):
        features= OrgansInSlicesFeatures(self.basePath)
        x,y=features.Prepare(self.databasePath,2,368,368,1.50)
        self.assertEqual(len(x), 2)
        self.assertEqual(len(y), 2)
        self.assertEqual(x[0].shape, (368,368))
        self.assertEqual(y[0].shape, (368,368,1))
        self.assertEqual(x[1].shape, (368,368))
        self.assertEqual(y[1].shape, (368,368,1))
        ShowMask(x[0],"Image 0")
        ShowMask(y[0],"Mask 0") 
        ShowMask(x[1],"Image 1")
        ShowMask(y[1],"Mask 1") 
    
    def test_ICanPrepare100Cases(self):
        features= OrgansInSlicesFeatures(self.basePath)
        x,y=features.Prepare(self.databasePath,100,368,368,1.50)
        self.assertEqual(len(x), 100)
        self.assertEqual(len(y), 100)
        self.assertEqual(x[0].shape, (368,368))
        self.assertEqual(y[0].shape, (368,368,1))
        self.assertEqual(x[99].shape, (368,368))
        self.assertEqual(y[99].shape, (368,368,1))
        ShowMask(x[0],"Image 0")
        ShowMask(y[0],"Mask 0") 
        ShowMask(x[99],"Image 1")
        ShowMask(y[99],"Mask 1") 


if __name__ == '__main__':
    unittest.main()
    