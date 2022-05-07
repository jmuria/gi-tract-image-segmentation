import glob
from OrgansInSlicesData import OrgansInSlicesData
from OrgansInSlicesMasks import OrgansInSlicesMasks
from ScanImage import ScanImage



class OrgansInSlicesFeatures:
    basePath='..\\input\\uw-madison-gi-tract-image-segmentation\\train\\'

    
    def Prepare(self,numSamples, height, width,pixelSize):
        x=[]
        y=[]
        maskData=OrgansInSlicesData.PrepareImageDataFromDatabase()
        for i, row in maskData.iloc[:numSamples].iterrows():    
            filePath=self.CreatePath(row['num_case'],row['day'],row['num_slice'])            
            sampleWidth,sampleHeight,samplePixelSize=OrgansInSlicesFeatures.GetSizesFromPath(filePath)
            image=ScanImage.Create(filePath) 
        
            if samplePixelSize!=pixelSize:
                image=ScanImage.ConvertPixelSize(image,samplePixelSize,pixelSize)

        
            maskImages,maskClasses=OrgansInSlicesMasks.CreateMasks(maskData,row['num_case'],row['day'],row['num_slice'],sampleHeight,sampleWidth)

            if(height!=sampleHeight | width!= sampleWidth ):
                 image=ScanImage.ResizeWithoutScaling(image,height,width)
                 for i,maskimage in enumerate(maskImages):
                    maskImages[i]=ScanImage.ResizeWithoutScaling(maskimage,height,width)

            maskImage=OrgansInSlicesMasks.CreateCombinedMaskFromImages(maskImages,height,width)
            x.append(image)
            
            y.append(maskImage)
        return x,y
            
    
    def CreatePath(self,case,day,slice):
        SlicePath=self.basePath+'case'+str(case)+'\\case'+str(case)+'_day'+str(day) + '\\scans\\slice_'+str(slice).zfill(4)
        SlicesFilenamesList = glob.glob(SlicePath+"*")
        return SlicesFilenamesList[0]

    def GetSizesFromPath(slicePath):
        extensionSize=4
        pathParts=slicePath.split('\\')
        filename=pathParts[-1][:-extensionSize]
        imageAttrs=filename.split('_')
        width=int(imageAttrs[2])
        height=int(imageAttrs[3])
        pixelSize=float(imageAttrs[4])
        return width,height,pixelSize
