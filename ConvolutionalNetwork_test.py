import unittest
from  ScanImage import ScanImage
import matplotlib.pyplot as plt
import numpy as np
from ConvolutionalNetwork import ConvolutionalNetwork



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
        convNetwork.PrepareInput(300,300,1)
       
    


     def test_TheModelHasAnOutputOf300x300FourChannels(self):
        convNetwork=ConvolutionalNetwork()
        model=convNetwork.CreateModel()
        self.assertIsNotNone(model)
        convNetwork.PrepareInput(300,300,1)
        convNetwork.PrepareOutput(300,300,4)
        convNetwork.CompileModel()
        convNetwork.PlotModel()

      