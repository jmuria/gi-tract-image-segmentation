import pandas as pd


class OrgansInSlicesData:
    numCase=0



    organ_type_mapping = {
        'large_bowel': 0,
        'small_bowel': 1,
        'stomach': 2
        }

    def PrepareImageDataFromDatabase(databasePath):
        df = pd.read_csv(databasePath)
        mask_data = df[df['segmentation'].notnull()]
        mask_data['num_case']=mask_data['id'].map(lambda x: OrgansInSlicesData.RetrieveCaseIDFromID(x)) 
        mask_data['day']=mask_data['id'].map(lambda x: OrgansInSlicesData.RetrieveDayFromID(x)) 
        mask_data['num_slice']=mask_data['id'].map(lambda x: OrgansInSlicesData.RetrieveNumSliceFromID(x)) 
        return mask_data

    def RetrieveCaseIDFromID(CaseId):
        splits = CaseId.split('_')
        return int(splits[0][4:])


    def RetrieveDayFromID(CaseId):
        splits = CaseId.split('_')
        return int(splits[1][3:])

    def RetrieveNumSliceFromID(CaseId):
        splits = CaseId.split('_')
        return int(splits[3])


    def RetriveMaskInfo(maskData,NumCase,day,numSlice):
        #maskInfo = maskData[(maskData['num_case']==NumCase)
        #          &(maskData['day']==day)
        #          &(maskData['num_slice']==numSlice)]
        maskInfo= maskData.loc[(maskData['num_case']==NumCase) & (maskData['day']== day) & (maskData['num_slice']==numSlice),['class','segmentation']]
       
        return maskInfo
