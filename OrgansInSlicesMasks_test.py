import unittest
from  OrgansInSlicesMasks import OrgansInSlicesMasks
from OrgansInSlicesData import OrgansInSlicesData
import matplotlib.pyplot as plt
import numpy as np


def apply_mask(image, maskImage):  
        image = image / image.max()
        image = np.dstack((image, maskImage,image)) 
        return image

def ShowMask(maskImage,title):
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    ax.set_title(title+' ('+str(maskImage.shape[0])+','+str(maskImage.shape[1])+')')
    ax.imshow(maskImage)    
    plt.show()

class TestOrgansInSlicesMasks(unittest.TestCase):

   

    def test_ICanCreateAMaskImageFromACaseDayandSlice(self):
        maskData=OrgansInSlicesData.PrepareImageDataFromDatabase()
        maskImage,maskClasses=OrgansInSlicesMasks.CreateMasks(maskData,123,20,75,266,266)
        ShowMask(maskImage[0],"Mask")

    def test_ICanCreateAllTheMasksImageFromACaseDayandSlice(self):
        maskData=OrgansInSlicesData.PrepareImageDataFromDatabase()
        maskImage,maskClasses=OrgansInSlicesMasks.CreateMasks(maskData,123,20,75,266,266)
        self.assertEqual(len(maskImage), 2)
        self.assertEqual(len(maskClasses), 2)        
        self.assertEqual(maskClasses[0], "large_bowel")
        self.assertEqual(maskClasses[1], "stomach")        
        ShowMask(maskImage[0],"large_bowel")
        ShowMask(maskImage[1],"stomach")

    def test_ICanCreateAllTheMasksImageFromAnotherCaseDayandSlice(self):
        maskData=OrgansInSlicesData.PrepareImageDataFromDatabase()
        maskImage,maskClasses=OrgansInSlicesMasks.CreateMasks(maskData,9,22,73,360,310)
        self.assertEqual(len(maskImage), 3)
        self.assertEqual(len(maskClasses), 3)        
        self.assertEqual(maskClasses[0], "large_bowel")
        self.assertEqual(maskClasses[1], "small_bowel")
        self.assertEqual(maskClasses[2], "stomach")        
        ShowMask(maskImage[0],"large_bowel")
        ShowMask(maskImage[1],"small_bowel")
        ShowMask(maskImage[2],"stomach")




if __name__ == '__main__':
    unittest.main()