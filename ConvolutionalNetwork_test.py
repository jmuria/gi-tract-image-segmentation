import unittest
from  ScanImage import ScanImage
import matplotlib.pyplot as plt
import numpy as np
from ConvolutionalNetwork import ConvolutionalNetwork
from OrgansInSlicesData import OrgansInSlicesData
from OrgansInSlicesMasks import OrgansInSlicesMasks



class TestConvolutionalNetwork(unittest.TestCase):
     
     def test_ICanCreateAModel(self):
        convNetwork=ConvolutionalNetwork()
        model=convNetwork.CreateModel()
        self.assertIsNotNone(model)
        #self.assertEqual(image.shape[1], 266)
    
     def test_TheModelHasAnInputOf300x300OneChannel(self):
        convNetwork=ConvolutionalNetwork()
        model=convNetwork.CreateModel()
        self.assertIsNotNone(model)
        convNetwork.PrepareInput(360,360,1)
       
    


     def test_TheModelHasAnOutputOf300x300FourChannels(self):
        convNetwork=ConvolutionalNetwork()
        model=convNetwork.CreateModel()
        self.assertIsNotNone(model)
        convNetwork.PrepareInput(360,360,1)
        convNetwork.PrepareOutput(360,360,4)
        convNetwork.CompileModel()
        convNetwork.PlotModel()


     def test_ICanTrainTheModelWithTheExpectedImages(self):
        convNetwork=ConvolutionalNetwork()
        model=convNetwork.CreateModel()
        self.assertIsNotNone(model)
        convNetwork.PrepareInput(368,368,1)
        convNetwork.PrepareIntermediateFilters()
        convNetwork.PrepareOutput(368,368,4)
        convNetwork.CompileModel()
        convNetwork.PlotModel()

        filePath='..\\input\\uw-madison-gi-tract-image-segmentation\\train\\case101\\case101_day20\\scans\\slice_0001_266_266_1.50_1.50.png'
        image=ScanImage.Create(filePath) 
        image=ScanImage.ResizeWithoutScaling(image,368,368)
    
        maskData=OrgansInSlicesData.PrepareImageDataFromDatabase()
        maskImage=OrgansInSlicesMasks.CreateCombinedMask(maskData,101,20,1,368,368)
       
       
        convNetwork.Train([image],[maskImage],(368,368))
      

     def test_ICanTrainTheModelWithMoreThanOneCase(self):
        convNetwork=ConvolutionalNetwork()
        model=convNetwork.CreateModel()
        self.assertIsNotNone(model)
        convNetwork.PrepareInput(368,368,1)
        convNetwork.PrepareIntermediateFilters()
        convNetwork.PrepareOutput(368,368,4)
        convNetwork.CompileModel()
        convNetwork.PlotModel()

        images=[]
        filePath='..\\input\\uw-madison-gi-tract-image-segmentation\\train\\case101\\case101_day20\\scans\\slice_0001_266_266_1.50_1.50.png'
        image=ScanImage.Create(filePath) 
        image=ScanImage.ResizeWithoutScaling(image,368,368)
        images.append(image)

        filePath='..\\input\\uw-madison-gi-tract-image-segmentation\\train\\case43\\case43_day22\\scans\\slice_0082_266_266_1.50_1.50.png'
        image=ScanImage.Create(filePath) 
        image=ScanImage.ResizeWithoutScaling(image,368,368)
        images.append(image)
    
        maskOrgansImages=[]
        maskData=OrgansInSlicesData.PrepareImageDataFromDatabase()
        maskImage=OrgansInSlicesMasks.CreateCombinedMask(maskData,101,20,1,368,368)
        maskOrgansImages.append(maskImage)
        maskImage=OrgansInSlicesMasks.CreateCombinedMask(maskData,43,22,82,368,368)
        maskOrgansImages.append(maskImage)
       
       
        history=convNetwork.Train(images,maskOrgansImages,(368,368),batch_size=2,epochs=50)
       
        ConvolutionalNetwork.PlotHistory(history)
        