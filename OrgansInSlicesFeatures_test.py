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

    


    def test_ICanGetTheFullPathOfASampleFromData(self):
        features=OrgansInSlicesFeatures()
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
        features= OrgansInSlicesFeatures()
        x,y=features.Prepare(1,368,368,1.50)
        self.assertEqual(len(x), 1)
        self.assertEqual(len(y), 1)
        self.assertEqual(x[0].shape, (368,368))
        self.assertEqual(y[0].shape, (368,368,1))
        ShowMask(x[0],"Image 0")
        ShowMask(y[0],"Mask 0") 


if __name__ == '__main__':
    unittest.main()
    