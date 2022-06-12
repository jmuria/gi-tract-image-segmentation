from OrgansInSlicesFeatures import OrgansInSlicesFeatures
from ConvolutionalNetwork import ConvolutionalNetwork
from OrgansInSlicesTestData import OrgansInSlicesTestData
from OrgansInSlicesMasks import OrgansInSlicesMasks
import numpy as np
from tensorflow.keras.utils import to_categorical

trainBasePath='../input/uw-madison-gi-tract-image-segmentation/train/'
databasePath='../input/uw-madison-gi-tract-image-segmentation/train.csv'
testBasePath='../test/'
resultDatabasePath='../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv'
modelPath='../output/mymodel.h5'
trainModel=True
numClasses=3
numSamplesToTrain=20
numEpocs=10


features= OrgansInSlicesFeatures(trainBasePath)
x,y=features.Prepare(databasePath,numSamplesToTrain,368,368,1.50)



train_masks_cat = to_categorical(y, num_classes=numClasses)
y_train_cat = train_masks_cat.reshape((y.shape[0], y.shape[1], y.shape[2], numClasses))

from sklearn.model_selection import train_test_split
X1, X_test, y1, y_test = train_test_split(x, y_train_cat, test_size = 0.90, random_state = 0)

convNetwork=ConvolutionalNetwork()


model=convNetwork.CreateModel()

if(trainModel):
    convNetwork.PrepareInput(368,368,1)
    convNetwork.PrepareIntermediateFilters()
    convNetwork.PrepareOutput(368,368,numClasses)
    convNetwork.CompileModel()
    convNetwork.PlotModel()



    history=convNetwork.Train(x,y_train_cat,(368,368),batch_size=5,epochs=numEpocs,num_classes=numClasses)
        
    ConvolutionalNetwork.PlotHistory(history)
    convNetwork.SaveModel(modelPath)
else:
    convNetwork.LoadModel(modelPath)

organsTestData=OrgansInSlicesTestData(testBasePath)
imagePathList=organsTestData.FindFiles()
convertedPaths=[]
for path in imagePathList:
    convertedPaths.append(path.replace('\\','/'))
testImages=OrgansInSlicesTestData.PrepareImages(convertedPaths,368,368,1.50)

X_testTemp=[]
X_testTemp.append(X_test[0])
X_testTemp.append(X_test[1])
X_test=np.array(X_testTemp)
Y_testTemp=[]
Y_testTemp.append(y_test[0])
Y_testTemp.append(y_test[1])
y_test=np.array(Y_testTemp)
Predictions=convNetwork.Predict(X_test,(368,368)) #testImages,(368,368))


y_pred_argmax=np.argmax(Predictions, axis=3)


from keras.metrics import MeanIoU
IOU_keras = MeanIoU(num_classes=numClasses)  
IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
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
OrgansInSlicesMasks.ShowMask(y_pred_argmax[0],"Prediction 1")
OrgansInSlicesMasks.ShowMask(y[0],"Mask 1")
OrgansInSlicesMasks.ShowMask(X_test[1],"Sclice 2")
OrgansInSlicesMasks.ShowMask(y_pred_argmax[1],"Prediction 2")
OrgansInSlicesMasks.ShowMask(y[1],"Mask2")

#resultDatabase=OrgansInSlicesTestData.CreateResultDatabase(resultDatabasePath,convertedPaths,y_pred_argmax,368,368,1.50)
#resultDatabase.to_csv('../output/submission.csv',index=False )