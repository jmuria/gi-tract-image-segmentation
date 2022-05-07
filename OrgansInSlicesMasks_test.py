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

   




 

    def test_ICanCreateAllTheMasksImageFromACaseDayandSlice(self):
        maskData=OrgansInSlicesData.PrepareImageDataFromDatabase()
        maskImage,maskClasses=OrgansInSlicesMasks.CreateMasks(maskData,123,20,75,266,266)
        self.assertEqual(len(maskImage), 4)
        self.assertEqual(len(maskClasses), 2)        
        self.assertEqual(maskClasses[0], "large_bowel")
        self.assertEqual(maskClasses[1], "stomach")        
        ShowMask(maskImage[0],"Background")
        ShowMask(maskImage[1],"large_bowel")
        ShowMask(maskImage[2],"small_bowel")
        ShowMask(maskImage[3],"stomach")
  
    def test_ICanCreateAllTheMasksImageFromAnotherCaseDayandSlice(self):
        maskData=OrgansInSlicesData.PrepareImageDataFromDatabase()
        maskImage,maskClasses=OrgansInSlicesMasks.CreateMasks(maskData,9,22,73,360,310)
        self.assertEqual(len(maskImage), 4)
        self.assertEqual(len(maskClasses), 3)        
        self.assertEqual(maskClasses[0], "large_bowel")
        self.assertEqual(maskClasses[1], "small_bowel")
        self.assertEqual(maskClasses[2], "stomach")        
        ShowMask(maskImage[0],"Background")
        ShowMask(maskImage[1],"large_bowel")
        ShowMask(maskImage[2],"small_bowel")
        ShowMask(maskImage[3],"stomach")

    def test_ICanCreateACombinedMaskFromACaseDayandSlice(self):
        maskData=OrgansInSlicesData.PrepareImageDataFromDatabase()
        maskImage=OrgansInSlicesMasks.CreateCombinedMask(maskData,9,22,73,360,310)
        self.assertEqual(maskImage.shape, (310,360,1))        
        ShowMask(maskImage,"Combined")
        print(maskImage)
        

        

    def test_ICanCorrectNoSquareMasks(self):
        maskData=OrgansInSlicesData.PrepareImageDataFromDatabase()
        maskImages,maskClasses=OrgansInSlicesMasks.CreateMasks(maskData,9,22,73,360,310)
        self.assertEqual(maskImages[0].shape[0], 310)
        self.assertEqual(maskImages[0].shape[1], 360)
        maskImages=OrgansInSlicesMasks.CorrectNoSquareMasks(maskImages,maskImages[0].shape[0],maskImages[0].shape[1])
        self.assertEqual(maskImages[0].shape[0], 360)
        self.assertEqual(maskImages[0].shape[1], 360)
        ShowMask(maskImages[1],"large_bowel")
  

if __name__ == '__main__':
    unittest.main()