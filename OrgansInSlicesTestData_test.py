import unittest
import matplotlib.pyplot as plt
import numpy as np
from OrgansInSlicesTestData import OrgansInSlicesTestData
from OrgansInSlicesData import OrgansInSlicesData
from OrgansInSlicesMasks import OrgansInSlicesMasks

class TestOrgansInSlicesTestData(unittest.TestCase):
     
    def test_ICanFindAFileInTheTestFolder(self):
        
        organsTestData=OrgansInSlicesTestData()
        imagePathList=organsTestData.FindFiles()

        self.assertEqual(len(imagePathList),5)


    def test_ICanGetTheIdFromATestFile(self):
        testFile='..\\test\\case3\\case3_day4\\scans\\slice_0001_266_266_1.50_1.50.png'
        generatedID=OrgansInSlicesTestData.GetFileID(testFile)
        self.assertEqual(generatedID,'case3_day4_slice_0001')
    
    def test_ICanCreateTheDatabaseLinesWithATestFileAndTheMasks(self):
        maskData=OrgansInSlicesData.PrepareImageDataFromDatabase()
        maskImage=OrgansInSlicesMasks.CreateCombinedMask(maskData,9,22,73,360,310)

        testFiles=[]
        testFiles.append('..\\test\\case3\\case3_day4\\scans\\slice_0001_266_266_1.50_1.50.png')
        maskImages=[]
        maskImages.append(maskImage)
        resultDatabase=OrgansInSlicesTestData.CreateResultDatabase(testFiles,maskImages)
        self.assertEqual(len(resultDatabase.index),3)
        self.assertEqual(resultDatabase['id'].values[0],'case3_day4_slice_0001')
        self.assertEqual(resultDatabase['class'].values[0],'large_bowel')
        self.assertGreater(len(resultDatabase['predicted'].values[0]),0)
        self.assertEqual(resultDatabase['id'].values[1],'case3_day4_slice_0001')
        self.assertEqual(resultDatabase['class'].values[1],'small_bowel')
        self.assertGreater(len(resultDatabase['predicted'].values[1]),0)
        self.assertEqual(resultDatabase['id'].values[2],'case3_day4_slice_0001')
        self.assertEqual(resultDatabase['class'].values[2],'stomach')
        self.assertGreater(len(resultDatabase['predicted'].values[2]),0)
