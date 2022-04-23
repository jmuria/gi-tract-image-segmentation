import unittest
from  OrgansInSlicesData import OrgansInSlicesData
 
 
class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')



    def test_TheDatabaseHasSegmentation(self):
        maskData=OrgansInSlicesData.PrepareImageDataFromDatabase()
        self.assertTrue('segmentation' in maskData.columns)

    def test_ICanRetrieveTheNumCaseFromtheId(self):
        CaseID='case123_day20_slice_0002'
        self.assertEqual(OrgansInSlicesData.RetrieveCaseIDFromID(CaseID), 123)

    def test_ICanRetrieveTheDayFromtheId(self):
        CaseID='case123_day20_slice_0002'
        self.assertEqual(OrgansInSlicesData.RetrieveDayFromID(CaseID), 20)

    def test_ICanRetrieveTheNumSliceFromtheId(self):
        CaseID='case123_day20_slice_0002'
        self.assertEqual(OrgansInSlicesData.RetrieveNumSliceFromID(CaseID), 2)

    def test_TheDatabaseHasNumCase(self):
        maskData=OrgansInSlicesData.PrepareImageDataFromDatabase()
        #First element is case123_day20_slice_0065
        self.assertTrue('num_case' in maskData.columns)
        self.assertEqual(maskData['num_case'].iloc[:1].values, 123)


    def test_TheDatabaseHasCaseDay(self):
        maskData=OrgansInSlicesData.PrepareImageDataFromDatabase()
        #First element is case123_day20_slice_0065
        self.assertTrue('day' in maskData.columns)
        self.assertEqual(maskData['day'].iloc[:1].values, 20)

    def test_TheDatabaseHasSliceDay(self):
        maskData=OrgansInSlicesData.PrepareImageDataFromDatabase()
        #First element is case123_day20_slice_0065
        self.assertTrue('num_slice' in maskData.columns)
        self.assertEqual(maskData['num_slice'].iloc[:1].values, 65)

    def test_ICanRetrieveTheDifferentMaskInfoFromCaseNumDayAndSlice(self):
        maskData=OrgansInSlicesData.PrepareImageDataFromDatabase()
        maskInSlices=OrgansInSlicesData.RetriveMaskInfo(maskData,123,20,75)
        self.assertEqual(len(maskInSlices.index), 2)


if __name__ == '__main__':
    unittest.main()