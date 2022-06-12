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

    databasePath='../input/uw-madison-gi-tract-image-segmentation/train.csv'




 

    def test_ICanCreateAllTheMasksImageFromACaseDayandSlice(self):
        maskData=OrgansInSlicesData.PrepareImageDataFromDatabase(self.databasePath)
        maskImage,maskClasses=OrgansInSlicesMasks.CreateMasks(maskData,123,20,75,266,266)
        self.assertEqual(len(maskImage), 3)
        self.assertEqual(len(maskClasses), 2)        
        self.assertEqual(maskClasses[0], "large_bowel")
        self.assertEqual(maskClasses[1], "stomach")                
        ShowMask(maskImage[0],"large_bowel")
        ShowMask(maskImage[1],"small_bowel")
        ShowMask(maskImage[2],"stomach")
  
    def test_ICanCreateAllTheMasksImageFromAnotherCaseDayandSlice(self):
        maskData=OrgansInSlicesData.PrepareImageDataFromDatabase(self.databasePath)
        maskImage,maskClasses=OrgansInSlicesMasks.CreateMasks(maskData,9,22,73,360,310)
        self.assertEqual(len(maskImage), 3)
        self.assertEqual(len(maskClasses), 3)        
        self.assertEqual(maskClasses[0], "large_bowel")
        self.assertEqual(maskClasses[1], "small_bowel")
        self.assertEqual(maskClasses[2], "stomach")                
        ShowMask(maskImage[0],"large_bowel")
        ShowMask(maskImage[1],"small_bowel")
        ShowMask(maskImage[2],"stomach")

    def test_ICanCreateACombinedMaskFromACaseDayandSlice(self):
        maskData=OrgansInSlicesData.PrepareImageDataFromDatabase(self.databasePath)
        maskImage=OrgansInSlicesMasks.CreateCombinedMask(maskData,9,22,73,360,310)
        self.assertEqual(maskImage.shape, (310,360,1))        
        ShowMask(maskImage,"Combined")
        print(maskImage)
        

    def test_ICanExtractDifferentMasksFromACombinedMask(self):
        maskData=OrgansInSlicesData.PrepareImageDataFromDatabase(self.databasePath)
        maskImage=OrgansInSlicesMasks.CreateCombinedMask(maskData,9,22,73,360,310)
        self.assertEqual(maskImage.shape, (310,360,1))        
        ShowMask(maskImage,"Combined")
        maskArray=OrgansInSlicesMasks.ExtractMasks(maskImage,310,360)        
        ShowMask(maskArray[0],"large_bowel")
        ShowMask(maskArray[1],"small_bowel")
        ShowMask(maskArray[2],"stomach")

    def test_ICanCorrectNoSquareMasks(self):
        maskData=OrgansInSlicesData.PrepareImageDataFromDatabase(self.databasePath)
        maskImages,maskClasses=OrgansInSlicesMasks.CreateMasks(maskData,9,22,73,360,310)
        self.assertEqual(maskImages[0].shape[0], 310)
        self.assertEqual(maskImages[0].shape[1], 360)
        maskImages=OrgansInSlicesMasks.CorrectNoSquareMasks(maskImages,maskImages[0].shape[0],maskImages[0].shape[1])
        self.assertEqual(maskImages[0].shape[0], 360)
        self.assertEqual(maskImages[0].shape[1], 360)
        ShowMask(maskImages[1],"large_bowel")

    def test_ICanCreateTheRLEFromTheMaskImage(self):
        maskData=OrgansInSlicesData.PrepareImageDataFromDatabase(self.databasePath)
        maskInSlices=OrgansInSlicesData.RetriveMaskInfo(maskData,9,22,73)
        RLEFromData=maskInSlices['segmentation'].values[0]
        maskImages,maskClasses=OrgansInSlicesMasks.CreateMasks(maskData,9,22,73,360,310)
        OrgansInSlicesMasks.ShowMask(maskImages[0],"Mask0")
        createdRLE=OrgansInSlicesMasks.CreateRLEFromImage(maskImages[0])        
        self.assertEqual(createdRLE, RLEFromData)

if __name__ == '__main__':
    unittest.main()