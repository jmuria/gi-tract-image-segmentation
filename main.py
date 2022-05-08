from OrgansInSlicesFeatures import OrgansInSlicesFeatures
from ConvolutionalNetwork import ConvolutionalNetwork

features= OrgansInSlicesFeatures()
x,y=features.Prepare(100,368,368,1.50)

convNetwork=ConvolutionalNetwork()
model=convNetwork.CreateModel()
convNetwork.PrepareInput(368,368,1)
convNetwork.PrepareIntermediateFilters()
convNetwork.PrepareOutput(368,368,4)
convNetwork.CompileModel()
convNetwork.PlotModel()

history=convNetwork.Train(x,y,(368,368),batch_size=5,epochs=50)
       
ConvolutionalNetwork.PlotHistory(history)