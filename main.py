from OrgansInSlicesFeatures import OrgansInSlicesFeatures
from ConvNetwork2 import ConvNetwork2
from OrgansInSlicesTestData import OrgansInSlicesTestData
from OrgansInSlicesMasks import OrgansInSlicesMasks
import numpy as np
from tensorflow.keras.utils import to_categorical

trainBasePath='../input/uw-madison-gi-tract-image-segmentation/train/'
databasePath='../input/uw-madison-gi-tract-image-segmentation/train.csv'
testBasePath='../test/'
resultDatabasePath='../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv'
modelPath='../output/mymodel.h5'
submissionDatabasePath='../output/submission.csv'
trainModel=False
numClasses=3
numSamplesToTrain=10
numEpocs=10


def ShowIoU(numClasses,X_test,Predictions,y_test=None):
    
    if(y_test is not None):
        from keras.metrics import MeanIoU
        IOU_keras = MeanIoU(num_classes=numClasses)  
        IOU_keras.update_state(y_test[:,:,:,0], Predictions)
        print("Mean IoU =", IOU_keras.result().numpy())

        values = np.array(IOU_keras.get_weights()).reshape(numClasses, numClasses)
        print(values)
        class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] +  values[1,0]+ values[2,0])
        class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] +  values[0,1]+ values[2,1])
        class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] +  values[0,2]+ values[1,2])

        print("IoU for class1 is: ", class1_IoU)
        print("IoU for class2 is: ", class2_IoU)
        print("IoU for class3 is: ", class3_IoU)
        y_test=np.argmax(y_test, axis=3)

    OrgansInSlicesMasks.ShowMask(X_test[0],"Sclice 1")
    OrgansInSlicesMasks.ShowMask(Predictions[0],"Prediction 1")
    if(y_test is not None):
        OrgansInSlicesMasks.ShowMask(y_test[0],"Mask 1")
    OrgansInSlicesMasks.ShowMask(X_test[1],"Sclice 2")
    OrgansInSlicesMasks.ShowMask(Predictions[1],"Prediction 2")
    if(y_test is not None):
        OrgansInSlicesMasks.ShowMask(y_test[1],"Mask2")


features= OrgansInSlicesFeatures(trainBasePath)
x,y=features.Prepare(databasePath,numSamplesToTrain,368,368,1.50)



train_masks_cat = to_categorical(y, num_classes=numClasses)
y_train_cat = train_masks_cat.reshape((y.shape[0], y.shape[1], y.shape[2], numClasses))

from sklearn.model_selection import train_test_split
X1, X_test, y1, y_test = train_test_split(x, y_train_cat, test_size = 0.20, random_state = 0)

convNetwork=ConvNetwork2()


model=convNetwork.CreateModel()

if(trainModel):
    convNetwork.PrepareInput(368,368,3)
    convNetwork.PrepareIntermediateFilters()
    convNetwork.PrepareOutput(368,368,numClasses)
    convNetwork.CompileModel()
    convNetwork.PlotModel()



    history=convNetwork.Train(x,y_train_cat,(368,368),batch_size=2,epochs=numEpocs,num_classes=numClasses)
        
    ConvNetwork2.PlotHistory(history)
    convNetwork.SaveModel(modelPath)
else:
    convNetwork.LoadModel(modelPath)

organsTestData=OrgansInSlicesTestData(testBasePath)
imagePathList=organsTestData.FindFiles()
convertedPaths=[]
for path in imagePathList:
    convertedPaths.append(path.replace('\\','/'))
testImages=OrgansInSlicesTestData.PrepareImages(convertedPaths,368,368,1.50)




Predictions=None
for testImage in testImages:
    arrayofTestImages=[]
    arrayofTestImages.append(testImage)
    arrayofTestImages=np.array(arrayofTestImages)
    Prediction=convNetwork.Predict(arrayofTestImages,(368,368))   
    if Predictions is None:
        Predictions=Prediction
    else:
        Predictions=np.concatenate((Predictions, Prediction), axis=0)        
  

Predictions=np.argmax(Predictions, axis=3)
ShowIoU(numClasses,testImages,Predictions)


resultDatabase=OrgansInSlicesTestData.CreateResultDatabase(resultDatabasePath,convertedPaths,Predictions,368,368,1.50)

print("Saving database",submissionDatabasePath)
resultDatabase.to_csv(submissionDatabasePath,index=False )