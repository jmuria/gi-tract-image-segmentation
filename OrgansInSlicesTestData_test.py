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
     
    basePath='../test/'
    databasePath='../input/uw-madison-gi-tract-image-segmentation/train.csv'
    resultDatabasePath='../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv'

    def test_ICanFindAFileInTheTestFolder(self):
        
        organsTestData=OrgansInSlicesTestData(self.basePath)
        imagePathList=organsTestData.FindFiles()

        self.assertEqual(len(imagePathList),5)

    def test_ICanPrepareTheTestImages(self):
        
        organsTestData=OrgansInSlicesTestData(self.basePath)
        imagePathList=organsTestData.FindFiles()
        convertedPaths=[]
        for path in imagePathList:
            convertedPaths.append(path.replace('\\','/'))
        testImages=OrgansInSlicesTestData.PrepareImages(convertedPaths,368,368,1.50)
        self.assertEqual(len(testImages),5)
        ShowImage(testImages[0],"Image 0")
        ShowImage(testImages[1],"Image 0")
        ShowImage(testImages[4],"Image 4")



    def test_ICanGetTheIdFromATestFile(self):
        testFile='../test/case3/case3_day4/scans/slice_0001_266_266_1.50_1.50.png'
        generatedID=OrgansInSlicesTestData.GetFileID(testFile)
        self.assertEqual(generatedID,'case3_day4_slice_0001')
    

    def test_ICanCreateAnEmptyDatabase(self):
        testFiles=[]
        testFiles.append('../test/case3/case3_day4/scans/slice_0001_266_266_1.50_1.50.png')
        resultDatabase=OrgansInSlicesTestData.CreateEmptyDatabase(self.resultDatabasePath,testFiles)
        self.assertEqual(len(resultDatabase.index),3)
        

    def test_ICanCreateTheDatabaseLinesWithATestFileAndTheMasks(self):
        maskData=OrgansInSlicesData.PrepareImageDataFromDatabase(self.databasePath)
        maskImage=OrgansInSlicesMasks.CreateCombinedMask(maskData,9,22,73,360,310)

        testFiles=[]
        testFiles.append('../test/case3/case3_day4/scans/slice_0001_266_266_1.50_1.50.png')
        maskImages=[]
        maskImages.append(maskImage)
        resultDatabase=OrgansInSlicesTestData.CreateEmptyDatabase(self.resultDatabasePath,testFiles)
        resultDatabase=OrgansInSlicesTestData.UpdateResultDatabase(resultDatabase,testFiles[0],maskImages[0],368,368,1.50)
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
        maskData=OrgansInSlicesData.PrepareImageDataFromDatabase(self.databasePath)
        maskImage=OrgansInSlicesMasks.CreateCombinedMask(maskData,9,22,73,360,310)

        testFiles=[]
        testFiles.append('../test/case3/case3_day4/scans/slice_0001_266_266_1.50_1.50.png')
        testFiles.append('../test/case4/case4_day3/scans/slice_0001_266_266_1.50_1.50.png')
        maskImages=[]
        maskImages.append(maskImage)
        maskImages.append(maskImage)
        resultDatabase=OrgansInSlicesTestData.CreateEmptyDatabase(self.resultDatabasePath,testFiles)
        resultDatabase=OrgansInSlicesTestData.UpdateResultDatabase(resultDatabase,testFiles[0],maskImages[0],368,368,1.50)
        resultDatabase=OrgansInSlicesTestData.UpdateResultDatabase(resultDatabase,testFiles[1],maskImages[1],368,368,1.50)
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

    def test_ICanSaveTheDatabas(self):
        outputDatabasePath='../output/submissionTest.csv'
        testFiles=[]
        testFiles.append('../test/case3/case3_day4/scans/slice_0001_266_266_1.50_1.50.png')
        testFiles.append('../test/case4/case4_day3/scans/slice_0001_266_266_1.50_1.50.png')
        resultDatabase=OrgansInSlicesTestData.CreateEmptyDatabase(self.resultDatabasePath,testFiles)
        self.assertEqual(len(resultDatabase.index),6)     
        OrgansInSlicesTestData.SaveDatabase(resultDatabase,outputDatabasePath)
        from os.path import exists

        self.assertTrue(exists(outputDatabasePath))


        