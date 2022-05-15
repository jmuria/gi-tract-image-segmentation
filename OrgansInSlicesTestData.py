from glob import glob
import pandas as pd
from OrgansInSlicesFeatures import OrgansInSlicesFeatures
from ScanImage import ScanImage
from OrgansInSlicesMasks import OrgansInSlicesMasks
from OrgansInSlicesData import OrgansInSlicesData

class OrgansInSlicesTestData:
   

    def __init__(self, basePath):
        self.basePath = basePath
    
    def FindFiles(self):
        fullPath=self.basePath+'*/*/scans/*.png'
        return  glob(fullPath)

    def PrepareImages(filePaths,expectedWidth,expectedHeight,expectedPixelSize):
        images=[]
        for path in filePaths:
           image,sampleWidth,sampleHeight,samplePixelSize=OrgansInSlicesFeatures.CreateSampleImage(path)
           if samplePixelSize!=expectedPixelSize:
                 image=ScanImage.ConvertPixelSize(image,samplePixelSize,expectedPixelSize)
            
           if(expectedHeight!=sampleHeight | expectedWidth!= sampleWidth ):
                image=ScanImage.ResizeWithoutScaling(image,expectedHeight,expectedWidth)   
           images.append(image)

        return images

    def GetFileID(imagePath):
        parts=imagePath.split('/')
        caseAndDay=parts[-3]
        caseParts=caseAndDay.split('_')
        case=int(caseParts[0][4:])
        day=int(caseParts[1][3:])
        fileNameParts=parts[-1].split('_')
        slice=int(fileNameParts[1])
        fileID='case'+str(case)+'_day'+str(day)+'_slice_'+str(slice).zfill(4)
        return fileID
    
    def CreateResultDatabase(submissionFilePath,filePaths,maskImages,width,height,pixelSize):
        
        submission_data = pd.read_csv(submissionFilePath)
        submission_data =submission_data[0:0]
        i=0
        for path in filePaths:
            id=OrgansInSlicesTestData.GetFileID(path)
            sampleWidth,sampleHeight,samplePixelSize=OrgansInSlicesFeatures.GetSizesFromPath(path)
            composedMask=maskImages[i]
            if samplePixelSize!=pixelSize:
                composedMask=ScanImage.ConvertPixelSize(composedMask,pixelSize,samplePixelSize)
            
            if(height!=sampleHeight | width!= sampleWidth ):
                composedMask=ScanImage.ResizeWithoutScaling(composedMask,sampleHeight,sampleWidth) 

      
            maskArray=OrgansInSlicesMasks.ExtractMasks(composedMask,sampleWidth,sampleHeight)
            
            for organType in OrgansInSlicesData.organ_type_mapping:
                                           
                rleInfo=OrgansInSlicesMasks.CreateRLEFromImage(maskArray[OrgansInSlicesData.organ_type_mapping[organType]])  
                df2 = {'id': id, 'class': organType,'predicted': rleInfo}
                submission_data = submission_data.append(df2, ignore_index = True)
               
            
            i=i+1
        
        return submission_data