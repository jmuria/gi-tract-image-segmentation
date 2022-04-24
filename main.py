import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd
from PIL import Image
from skimage import color
from glob import glob
from OrgansInSlicesData import OrgansInSlicesData
from OrgansInSlicesMasks import OrgansInSlicesMasks
from ScanImage import ScanImage
import warnings




def PrepareImageDataFromFolder():
  
  list_images = glob('..\\input\\uw-madison-gi-tract-image-segmentation\\train\\*\\*\\scans\\*.png')
  image_details = pd.DataFrame({'Path':list_images})
  splits = image_details['Path'].str.split("\\", n = 7, expand = True)
  image_details['Case_no_And_Day'] = splits[5]
  image_details['Slice_Info'] = splits[7]


  splits = image_details['Case_no_And_Day'].str.split("_", n = 2, expand = True)
  image_details['Case_no'] = splits[0].str[4:].astype(int)
  image_details['Day'] = splits[1].str[3:].astype(int)


  splits = image_details['Slice_Info'].str.split("_", n = 6, expand = True)
  image_details['Slice_no'] = splits[1].astype(int)
  image_details['Height'] = splits[2].astype(int)
  image_details['Width'] = splits[3].astype(int)
  image_details['Pix_Heigh'] = splits[4].astype(float)
  image_details['Pix_Width'] = splits[5].str[:-4].astype(float)
  return image_details





image_details=PrepareImageDataFromFolder()
mask_data=OrgansInSlicesData.PrepareImageDataFromDatabase()
index_list = list(mask_data.index)

def ShowSomeImages(numImages,imageList):
  if numImages>0:
    plt.subplots(figsize=(10,15))
  for i in range(numImages):
    index = np.random.randint(0,imageList.shape[0])
    image = Image.open(imageList.loc[index, 'Path'])
    image = np.array(image)

    plt.subplot(4, 3, i + 1)

    title = (imageList.loc[index, 'Case_no_And_Day'] + 
            '_Slice_no_' + str(imageList.loc[index, 'Slice_no']))

    plt.title(title)
    plt.imshow(np.interp(image, [np.min(image), np.max(image)], [0,255]))
    # plt.imshow(image / image.max())  #This will also serve the purpose.

ShowSomeImages(0,image_details)







def CreateOrganTypesList(dataset):
    organ_type_mapping = {
        'large_bowel': 1,
        'small_bowel': 2,
        'stomach': 3
        }
    dict_organ_type = {}
    for sample_id, organ_type in dataset[['id', 'class']].values:
        dict_organ_type[sample_id] = organ_type
    
    dict_organ_type_encoded = {sample_id: organ_type_mapping[organ_type] for sample_id, organ_type in dict_organ_type.items()}

    return dict_organ_type_encoded
dict_organ_type=CreateOrganTypesList(mask_data)



def apply_mask(image, maskImage):  
  image = image / image.max()
  image = np.dstack((image, maskImage,image)) 
  return image

def CorrectImagesDimensions(width,height,image):
    added = height-width        
    image = np.concatenate((np.zeros((added,height)),image),axis=0)    
    return image



def ShowImages(image,maskImage,applyMerge):
  fig, ax = plt.subplots(1,3, figsize=(10,10))
  ax[0].set_title('Image ('+str(image.shape[0])+','+str(image.shape[1])+')')
  ax[0].imshow(image)
  ax[1].set_title('Mask')    
  ax[1].imshow(maskImage)
  if applyMerge:
    ax[2].set_title('Merged')
    ax[2].imshow(apply_mask(image, maskImage))
  plt.show()

def PrepareImageAndMask(numCase,day,numSlice,mask_data,image_details):
    
  x = image_details[(image_details['Case_no']==numCase)
                  &(image_details['Day']==day)
                  &(image_details['Slice_no']==numSlice)]


  image=ScanImage.Create(path=x['Path'].values[0])  
    
  maskImage,maskClasses=OrgansInSlicesMasks.CreateMasks(mask_data,numCase,day,numSlice,image.shape[1],image.shape[0])
 
  if x.Height.values[0]!=x.Width.values[0]:   
      image=CorrectImagesDimensions(x.Width.values[0],x.Height.values[0],image) 
      maskImage=OrgansInSlicesMasks.CorrectNoSquareMasks(maskImage,x.Width.values[0], x.Height.values[0])       
  return image,maskImage




for i in range(1):
  #index = index_list[np.random.randint(0,len(index_list) - 1)]
  #curr_organ_data=mask_data.loc[index]
  #image,maskImage=PrepareImageAndMask(curr_organ_data['num_case'],curr_organ_data['day'],curr_organ_data['num_slice'],mask_data,image_details)
  image,maskImage=PrepareImageAndMask(9,22,73,mask_data,image_details)
  ShowImages(image,maskImage[0],True)
  ShowImages(image,maskImage[1],True)
  ShowImages(image,maskImage[2],True)
  
plt.show()