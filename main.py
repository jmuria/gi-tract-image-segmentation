from OrgansInSlicesFeatures import OrgansInSlicesFeatures
from ConvolutionalNetwork import ConvolutionalNetwork
from OrgansInSlicesTestData import OrgansInSlicesTestData
from OrgansInSlicesMasks import OrgansInSlicesMasks

trainBasePath='..\\input\\uw-madison-gi-tract-image-segmentation\\train\\'

features= OrgansInSlicesFeatures(trainBasePath)
x,y=features.Prepare(100,368,368,1.50)

convNetwork=ConvolutionalNetwork()
model=convNetwork.CreateModel()
convNetwork.PrepareInput(368,368,1)
convNetwork.PrepareIntermediateFilters()
convNetwork.PrepareOutput(368,368,4)
convNetwork.CompileModel()
convNetwork.PlotModel()

history=convNetwork.Train(x,y,(368,368),batch_size=5,epochs=2)
       
ConvolutionalNetwork.PlotHistory(history)

organsTestData=OrgansInSlicesTestData()
imagePathList=organsTestData.FindFiles()
testImages=OrgansInSlicesTestData.PrepareImages(imagePathList,368,368,1.50)
Predictions=convNetwork.Predict(testImages,(368,368))

resultDatabase=OrgansInSlicesTestData.CreateResultDatabase(testImages,Predictions,368,368,1.50)
resultDatabase.to_csv('../output/submission.csv')  