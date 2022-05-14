from glob import glob
import pandas as pd

class OrgansInSlicesTestData:
    basePath='..\\test\\'

    
    def FindFiles(self):
        fullPath=self.basePath+'*\\*\\scans\\*.png'
        return  glob(fullPath)

    def GetFileID(imagePath):
        parts=imagePath.split('\\')
        caseAndDay=parts[-3]
        caseParts=caseAndDay.split('_')
        case=int(caseParts[0][4:])
        day=int(caseParts[1][3:])
        fileNameParts=parts[-1].split('_')
        slice=int(fileNameParts[1])
        fileID='case'+str(case)+'_day'+str(day)+'_slice_'+str(slice).zfill(4)
        return fileID
    
    def CreateResultDatabase(filePaths,maskImages):
        
        submission_data = pd.read_csv('..\\input\\uw-madison-gi-tract-image-segmentation\\sample_submission.csv')
        submission_data =submission_data[0:0]
        for path in filePaths:
            id=OrgansInSlicesTestData.GetFileID(path)

            df2 = {'id': id, 'class': 'large_bowel','predicted': '55 55'}
            submission_data = submission_data.append(df2, ignore_index = True)

            df2 = {'id': id, 'class': 'small_bowel','predicted': '55 55'}
            submission_data = submission_data.append(df2, ignore_index = True)

            df2 = {'id': id, 'class': 'stomach','predicted': '55 55'}
            submission_data = submission_data.append(df2, ignore_index = True)
        
        return submission_data