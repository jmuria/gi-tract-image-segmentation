import unittest
from  OrgansInSlicesMasks import OrgansInSlicesMasks
from OrgansInSlicesData import OrgansInSlicesData
import matplotlib.pyplot as plt
import numpy as np


def apply_mask(image, maskImage):  
        image = image / image.max()
        image = np.dstack((image, maskImage,image)) 
        return image

def ShowMask(maskImage):
    fig, ax = plt.subplots(1,3, figsize=(10,10))
    ax[0].set_title('Mask ('+str(maskImage.shape[0])+','+str(maskImage.shape[1])+')')
    ax[0].imshow(maskImage)    
    plt.show()

class TestOrgansInSlicesMasks(unittest.TestCase):

   

    def test_ICanCreateAMaskImageFromACaseDayandSlice(self):
        maskData=OrgansInSlicesData.PrepareImageDataFromDatabase()
        maskImage=OrgansInSlicesMasks.CreateMasks(maskData,123,20,75,266,266)
        ShowMask(maskImage)




if __name__ == '__main__':
    unittest.main()