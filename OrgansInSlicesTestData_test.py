import unittest
import matplotlib.pyplot as plt
import numpy as np
from OrgansInSlicesTestData import OrgansInSlicesTestData
from OrgansInSlicesData import OrgansInSlicesData
from OrgansInSlicesMasks import OrgansInSlicesMasks

def ShowImage(maskImage,title):
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    ax.set_title(title+' ('+str(maskImage.shape[0])+','+str(maskImage.shape[1])+')')
    ax.imshow(maskImage)    
    plt.show()

class TestOrgansInSlicesTestData(unittest.TestCase):
     
    def test_ICanFindAFileInTheTestFolder(self):
        
        organsTestData=OrgansInSlicesTestData()
        imagePathList=organsTestData.FindFiles()

        self.assertEqual(len(imagePathList),5)

    def test_ICanPrepareTheTestImages(self):
        
        organsTestData=OrgansInSlicesTestData()
        imagePathList=organsTestData.FindFiles()
        testData=OrgansInSlicesTestData()
        testImages=OrgansInSlicesTestData.PrepareImages(imagePathList,368,368,1.50)
        self.assertEqual(len(testImages),5)
        ShowImage(testImages[0],"Image 0")
        ShowImage(testImages[1],"Image 0")
        ShowImage(testImages[4],"Image 4")



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
        resultDatabase=OrgansInSlicesTestData.CreateResultDatabase(testFiles,maskImages,368,368,1.50)
        self.assertEqual(len(resultDatabase.index),3)
        self.assertEqual(resultDatabase['id'].values[0],'case3_day4_slice_0001')
        self.assertTrue('stomach' in resultDatabase['class'].values)
        self.assertGreater(len(resultDatabase['predicted'].values[0]),0)
        self.assertEqual(resultDatabase['id'].values[1],'case3_day4_slice_0001')
        self.assertTrue('large_bowel' in resultDatabase['class'].values)
        self.assertGreater(len(resultDatabase['predicted'].values[1]),0)
        self.assertEqual(resultDatabase['id'].values[2],'case3_day4_slice_0001')        
        self.assertTrue('small_bowel' in resultDatabase['class'].values)
        self.assertGreater(len(resultDatabase['predicted'].values[2]),0)


    def test_ICanCreateTheDatabaseLinesWithTwoTestFilesAndTheMasks(self):
        maskData=OrgansInSlicesData.PrepareImageDataFromDatabase()
        maskImage=OrgansInSlicesMasks.CreateCombinedMask(maskData,9,22,73,360,310)

        testFiles=[]
        testFiles.append('..\\test\\case3\\case3_day4\\scans\\slice_0001_266_266_1.50_1.50.png')
        testFiles.append('..\\test\\case4\\case4_day3\\scans\\slice_0001_266_266_1.50_1.50.png')
        maskImages=[]
        maskImages.append(maskImage)
        maskImages.append(maskImage)
        resultDatabase=OrgansInSlicesTestData.CreateResultDatabase(testFiles,maskImages,368,368,1.50)
        self.assertEqual(len(resultDatabase.index),6)        
        self.assertTrue('stomach' in resultDatabase['class'].values)
        self.assertTrue('case3_day4_slice_0001' in resultDatabase['id'].values)
        self.assertGreater(len(resultDatabase['predicted'].values[0]),0)
        self.assertEqual(resultDatabase['id'].values[1],'case3_day4_slice_0001')
        self.assertTrue('large_bowel' in resultDatabase['class'].values)
        self.assertGreater(len(resultDatabase['predicted'].values[1]),0)
        self.assertEqual(resultDatabase['id'].values[2],'case3_day4_slice_0001')        
        self.assertTrue('small_bowel' in resultDatabase['class'].values)
        self.assertGreater(len(resultDatabase['predicted'].values[2]),0)
        self.assertTrue('case4_day3_slice_0001' in resultDatabase['id'].values)