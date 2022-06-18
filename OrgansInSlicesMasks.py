import matplotlib.pyplot as plt
import numpy as np
from OrgansInSlicesData import OrgansInSlicesData

class OrgansInSlicesMasks:
    noSquareImagesOffset=50

    def get_pixel_loc(rle_string, width,height):
        rle = [int(i) for i in rle_string.split(' ')]
        pairs = list(zip(rle[0::2],rle[1::2]))

        # This for loop will help to understand better the above command.
        # pairs = []
        # for i in range(0, len(rle), 2):
        #   a.append((rle[i], rle[i+1])

        p_loc = []     #   Pixel Locations

        for start, length in pairs:
            for p_pos in range(start, start + length):
                p_loc.append((p_pos % width, p_pos // width))
        
        return p_loc
    
    def get_mask(mask, width,height):

        canvas = np.zeros([height,width]).T
        canvas[tuple(zip(*mask))] = 1

        # This is the Equivalent for loop of the above command for better understanding.
        # for pos in range(len(p_loc)):
        #   canvas[pos[0], pos[1]] = 1

        return canvas.T
    
    # read graysclae img
    # Thanks to q-viper https://q-viper.github.io/2021/05/24/coding-run-length-encoding-in-python/
        
    def CreateRLEFromImage(img):
           # ref.: https://www.kaggle.com/stainsby/fast-tested-rle
   
        pixels = img.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] #+ 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)

    def CreateEmptyMask( width,height):

        return  np.zeros([width,height]).T        
    



    def ShowMask(maskImage,title):
        fig, ax = plt.subplots(1,1, figsize=(10,10))
        ax.set_title(title+' ('+str(maskImage.shape[0])+','+str(maskImage.shape[1])+')')
        ax.imshow(maskImage)    
        plt.show()

    def CreateMask(mask_data,width,height):   
        maskSegmentation=mask_data['segmentation'].values[0]
        p_loc = OrgansInSlicesMasks.get_pixel_loc(maskSegmentation, width,height)
        maskImage=OrgansInSlicesMasks.get_mask(p_loc,width,height)
        return maskImage



    def CreateMasks(maskData,numCase,day,numSlice,width,height):
        maskImage=[]
        maskInSlices=OrgansInSlicesData.RetriveMaskInfo(maskData,numCase,day,numSlice)
        maskClasses=maskInSlices['class'].values
     
        for organType in OrgansInSlicesData.organ_type_mapping:
            organIndex=maskInSlices.index[maskInSlices['class'] == organType].tolist()
            if( len(organIndex)>0):
                maskImage.append(OrgansInSlicesMasks.CreateMask(maskInSlices.loc[organIndex],width,height))
            else:
                maskImage.append(OrgansInSlicesMasks.CreateEmptyMask(width,height)) #Empty mask
        
        return  maskImage,maskClasses
    
    def CreateCombinedMask(maskData,numCase,day,numSlice,width,height):
        maskImages,maskClasses= OrgansInSlicesMasks.CreateMasks(maskData,numCase,day,numSlice,width,height)
        return OrgansInSlicesMasks.CreateCombinedMaskFromImages(maskImages,width,height)        
        

    def CreateCombinedMaskFromImages(maskImages,width,height):        
        combinedMaskImage = np.zeros((height,width) + (1,), dtype="uint8")
        increase=255/len(maskImages)
        for organIndex in range(len(OrgansInSlicesData.organ_type_mapping)):            
            combinedMaskImage = combinedMaskImage + np.expand_dims(maskImages[organIndex]*(organIndex), axis=2)
            combinedMaskImage[combinedMaskImage>organIndex]=organIndex+1  
                
        return  combinedMaskImage
    
    def ExtractMasks(maskImage,height,width):
        imageArray=[]        
      
        for organIndex in range(len(OrgansInSlicesData.organ_type_mapping)):
            mask = OrgansInSlicesMasks.CreateEmptyMask( width,height)
            mask[np.squeeze(maskImage==(organIndex+1))]=1
            imageArray.append(mask)
        return imageArray



    def CorrectMaskHeight(height,width,maskImage):
        added = width-height
        added2Mask=OrgansInSlicesMasks.noSquareImagesOffset    
        maskImage = np.concatenate((np.zeros((added2Mask,width)),maskImage),axis=0)   
        maskImage = np.concatenate((maskImage,np.zeros((added-added2Mask,width))),axis=0)   
        return maskImage

    def CorrectNoSquareMasks(maskImages,width,height):
        for index in range(len(maskImages)):
            maskImages[index]=OrgansInSlicesMasks.CorrectMaskHeight(width, height,maskImages[index])
        return maskImages