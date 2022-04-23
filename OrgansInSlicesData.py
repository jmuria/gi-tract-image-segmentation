import pandas as pd

class OrgansInSlicesData:
    numCase=0

    def PrepareImageDataFromDatabase():
        df = pd.read_csv('../input/uw-madison-gi-tract-image-segmentation/train.csv')
        mask_data = df[df['segmentation'].notnull()]
        mask_data['num_case']=mask_data['id'].map(lambda x: OrgansInSlicesData.RetrieveCaseIDFromID(x)) 
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
    
