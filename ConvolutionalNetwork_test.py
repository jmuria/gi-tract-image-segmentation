import unittest
from  ScanImage import ScanImage
import matplotlib.pyplot as plt
import numpy as np
import os 
from ConvolutionalNetwork import ConvolutionalNetwork
from OrgansInSlicesData import OrgansInSlicesData
from OrgansInSlicesMasks import OrgansInSlicesMasks
from OrgansInSlicesFeatures import OrgansInSlicesFeatures
from ConvolutionalNetwork import DiceLoss



class TestConvolutionalNetwork(unittest.TestCase):
     databasePath='../input/uw-madison-gi-tract-image-segmentation/train.csv'
     basePath='..\\input\\uw-madison-gi-tract-image-segmentation\\train\\'
     
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
        convNetwork.PrepareOutput(360,360,3)
        convNetwork.CompileModel()
        convNetwork.PlotModel()


     def test_ICanTrainTheModelWithTheExpectedImages(self):
        convNetwork=ConvolutionalNetwork()
        model=convNetwork.CreateModel()
        self.assertIsNotNone(model)
        convNetwork.PrepareInput(368,368,1)
        convNetwork.PrepareIntermediateFilters()
        convNetwork.PrepareOutput(368,368,3)
        convNetwork.CompileModel()
        convNetwork.PlotModel()

        filePath='..\\input\\uw-madison-gi-tract-image-segmentation\\train\\case101\\case101_day20\\scans\\slice_0001_266_266_1.50_1.50.png'
        image=ScanImage.Create(filePath) 
        image=ScanImage.ResizeWithoutScaling(image,368,368)
    
        maskData=OrgansInSlicesData.PrepareImageDataFromDatabase(self.databasePath)
        maskImage=OrgansInSlicesMasks.CreateCombinedMask(maskData,101,20,1,368,368)
       
       
        convNetwork.Train([image],[maskImage],(368,368))
      
     def test_ICanSaveAndRestoreAModel(self):
        convNetwork=ConvolutionalNetwork()
        model=convNetwork.CreateModel()
        self.assertIsNotNone(model)
        convNetwork.PrepareInput(368,368,1)
        convNetwork.PrepareIntermediateFilters()
        convNetwork.PrepareOutput(368,368,3)
        convNetwork.CompileModel()
        convNetwork.PlotModel()

        filePath='..\\input\\uw-madison-gi-tract-image-segmentation\\train\\case101\\case101_day20\\scans\\slice_0001_266_266_1.50_1.50.png'
        image=ScanImage.Create(filePath) 
        image=ScanImage.ResizeWithoutScaling(image,368,368)
    
        maskData=OrgansInSlicesData.PrepareImageDataFromDatabase(self.databasePath)
        maskImage=OrgansInSlicesMasks.CreateCombinedMask(maskData,101,20,1,368,368)
       
        modelPath='../output/_ICanSaveAndRestoreAModel.h5'
        convNetwork.Train([image],[maskImage],(368,368))
        convNetwork.SaveModel(modelPath)
        self.assertTrue(os.path.exists(modelPath))
        new_model = convNetwork.LoadModel(modelPath)
        self.assertIsNotNone(new_model)


     def test_ICanTrainTheModelWithMoreThanOneCase(self):
        convNetwork=ConvolutionalNetwork()
        model=convNetwork.CreateModel()
        self.assertIsNotNone(model)
        convNetwork.PrepareInput(368,368,1)
        convNetwork.PrepareIntermediateFilters()
        convNetwork.PrepareOutput(368,368,3)
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
        maskData=OrgansInSlicesData.PrepareImageDataFromDatabase(self.databasePath)
        maskImage=OrgansInSlicesMasks.CreateCombinedMask(maskData,101,20,1,368,368)
        maskOrgansImages.append(maskImage)
        maskImage=OrgansInSlicesMasks.CreateCombinedMask(maskData,43,22,82,368,368)
        maskOrgansImages.append(maskImage)
       
       
        history=convNetwork.Train(images,maskOrgansImages,(368,368),batch_size=2,epochs=50)
       
        ConvolutionalNetwork.PlotHistory(history)


     def test_ICanSetAnEarlyStop(self):
        convNetwork=ConvolutionalNetwork()
        model=convNetwork.CreateModel()
        self.assertIsNotNone(model)
        convNetwork.PrepareInput(368,368,1)
        convNetwork.PrepareIntermediateFilters()
        convNetwork.PrepareOutput(368,368,3)
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
        maskData=OrgansInSlicesData.PrepareImageDataFromDatabase(self.databasePath)
        maskImage=OrgansInSlicesMasks.CreateCombinedMask(maskData,101,20,1,368,368)
        maskOrgansImages.append(maskImage)
        maskImage=OrgansInSlicesMasks.CreateCombinedMask(maskData,43,22,82,368,368)
        maskOrgansImages.append(maskImage)
       
        patiente=4
        convNetwork.SetEarlyStop(patiente)
        history=convNetwork.Train(images,maskOrgansImages,(368,368),batch_size=2,epochs=50)
       
        ConvolutionalNetwork.PlotHistory(history)
        

     def test_ICanPredictWithATrainedModel(self):
        convNetwork=ConvolutionalNetwork()
        model=convNetwork.CreateModel()
        self.assertIsNotNone(model)
        convNetwork.PrepareInput(368,368,1)
        convNetwork.PrepareIntermediateFilters()
        convNetwork.PrepareOutput(368,368,3)
        convNetwork.CompileModel()
        convNetwork.PlotModel()

        features= OrgansInSlicesFeatures(self.basePath)
        x,y=features.Prepare(self.databasePath, 5,368,368,1.50)
        
        
   
        from tensorflow.keras.utils import to_categorical
        numpy_y=np.array(y)
        train_masks_cat = to_categorical(numpy_y, num_classes=3)
        y_train_cat = train_masks_cat.reshape((numpy_y.shape[0], numpy_y.shape[1], numpy_y.shape[2], 3))



        history=convNetwork.Train(x,y_train_cat,(368,368),batch_size=2,epochs=2,num_classes=3)
        ConvolutionalNetwork.PlotHistory(history)
        Predictions=convNetwork.Predict(x,(368,368))
        y_pred_argmax=np.argmax(Predictions, axis=3)


        from keras.metrics import MeanIoU
        n_classes = 3
        IOU_keras = MeanIoU(num_classes=n_classes)  
        IOU_keras.update_state(numpy_y[:,:,:,0], y_pred_argmax)
        print("Mean IoU =", IOU_keras.result().numpy())

        values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
        print(values)
        class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[1,0]+ values[2,0])
        class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[0,1]+ values[2,1])
        class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] +  values[0,2]+ values[1,2])
        
        print("IoU for class1 is: ", class1_IoU)
        print("IoU for class2 is: ", class2_IoU)
        print("IoU for class3 is: ", class3_IoU)        

        #preds_test_thresh = Predictions.astype(np.uint8)
        # Display a thresholded mask

        #test_img = preds_test_thresh[0, :, :, 0]

        #plt.imshow(test_img)


        OrgansInSlicesMasks.ShowMask(x[0],"Sclice 1")
        OrgansInSlicesMasks.ShowMask(y_pred_argmax[0],"Prediction 1")
        OrgansInSlicesMasks.ShowMask(y[0],"Mask 1")
        OrgansInSlicesMasks.ShowMask(x[1],"Sclice 2")
        OrgansInSlicesMasks.ShowMask(y_pred_argmax[1],"Prediction 2")
        OrgansInSlicesMasks.ShowMask(y[1],"Mask2")