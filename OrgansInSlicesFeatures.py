import glob
from OrgansInSlicesData import OrgansInSlicesData
from OrgansInSlicesMasks import OrgansInSlicesMasks
from ScanImage import ScanImage



class OrgansInSlicesFeatures:
    basePath='..\\input\\uw-madison-gi-tract-image-segmentation\\train\\'


    def HomogenizePixelSize(self,image,maskImages,pixelSize,expectedPixelSize):
        if pixelSize!=expectedPixelSize:
            image=ScanImage.ConvertPixelSize(image,pixelSize,expectedPixelSize)
            for i,maskimage in enumerate(maskImages):
                maskImages[i]=ScanImage.ConvertPixelSize(maskimage,pixelSize,expectedPixelSize)
        return image,maskImages

    def HomogenizeImageSize(self,image,maskImages,sampleHeight,expectedHeight,sampleWidth,expectedWidth):
        if(expectedHeight!=sampleHeight | expectedWidth!= sampleWidth ):
            image=ScanImage.ResizeWithoutScaling(image,expectedHeight,expectedWidth)
            for i,maskimage in enumerate(maskImages):
                maskImages[i]=ScanImage.ResizeWithoutScaling(maskimage,expectedHeight,expectedWidth)

        return image,maskImages

    def CreateSampleImage(self,row):        
        filePath=self.CreatePath(row['num_case'],row['day'],row['num_slice'])            
        sampleWidth,sampleHeight,samplePixelSize=OrgansInSlicesFeatures.GetSizesFromPath(filePath)
        image=ScanImage.Create(filePath) 
        return image,sampleWidth,sampleHeight,samplePixelSize

    def PrepareSingleSample(self,row,expectedHeight, expectedWidth,expectedPixelSize):        
        image,sampleWidth,sampleHeight,samplePixelSize=self.CreateSampleImage(row)
        maskImages,maskClasses=OrgansInSlicesMasks.CreateMasks(self.maskData,row['num_case'],row['day'],row['num_slice'],sampleHeight,sampleWidth)
        image,maskImages=self.HomogenizePixelSize(image,maskImages,samplePixelSize,expectedPixelSize)                
        image,maskImages=self.HomogenizeImageSize(image,maskImages,sampleHeight,expectedHeight,sampleWidth,expectedWidth) 
        maskImage=OrgansInSlicesMasks.CreateCombinedMaskFromImages(maskImages,expectedHeight,expectedWidth)
        return image,maskImage
    
    def Prepare(self,numSamples, height, width,pixelSize):
        x=[]
        y=[]
        self.maskData=OrgansInSlicesData.PrepareImageDataFromDatabase()
        for i, row in self.maskData.iloc[:numSamples].iterrows():    
            image,maskImage=self.PrepareSingleSample(row,height, width,pixelSize)            
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
