from OrgansInSlicesData import OrgansInSlicesData
import numpy as np

class OrgansInSlicesMasks:

    def get_pixel_loc(rle_string, img_shape):
        rle = [int(i) for i in rle_string.split(' ')]
        pairs = list(zip(rle[0::2],rle[1::2]))

        # This for loop will help to understand better the above command.
        # pairs = []
        # for i in range(0, len(rle), 2):
        #   a.append((rle[i], rle[i+1])

        p_loc = []     #   Pixel Locations

        for start, length in pairs:
            for p_pos in range(start, start + length):
                p_loc.append((p_pos % img_shape[1], p_pos // img_shape[0]))
        
        return p_loc
    
    def get_mask(mask, img_shape):

        canvas = np.zeros(img_shape).T
        canvas[tuple(zip(*mask))] = 1

        # This is the Equivalent for loop of the above command for better understanding.
        # for pos in range(len(p_loc)):
        #   canvas[pos[0], pos[1]] = 1

        return canvas.T

    def CreateMask(mask_data,width,height):   
        p_loc = OrgansInSlicesMasks.get_pixel_loc(mask_data.iloc(0)['segmentation'], width,height)
        maskImage=OrgansInSlicesMasks.get_mask(p_loc,width,height)
        return maskImage

    def CreateMasks(maskData,numCase,day,numSlice,width,height):
        maskInSlices=OrgansInSlicesData.RetriveMaskInfo(maskData,numCase,day,numSlice)
        maskImage=OrgansInSlicesMasks.CreateMask(maskInSlices,width,height)   