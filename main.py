import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd
from PIL import Image
from skimage import color
from glob import glob
import warnings




def PrepareData():
  df = pd.read_csv('../input/uw-madison-gi-tract-image-segmentation/train.csv')
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
  mask_data = df[df['segmentation'].notnull()]
  return image_details,mask_data


image_details,mask_data=PrepareData()
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




def get_pixel_loc(rle_string, img_shape):
  rle = [int(i) for i in rle_string.split(' ')]
  pairs = list(zip(rle[0::2],rle[1::2]))

  # This for loop will help to understand better the above command.
  # pairs = []
  # for i in range(0, len(rle), 2):
  #   a.append((rle[i], rle[i+1])

  p_loc = []     #   Pixel Locations

  for start, length in pairs:
    for p_pos in range(start, start + length):
      p_loc.append((p_pos % img_shape[1], p_pos // img_shape[0]))
  
  return p_loc



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

def get_mask(mask, img_shape):
  
  canvas = np.zeros(img_shape).T
  canvas[tuple(zip(*mask))] = 1

  # This is the Equivalent for loop of the above command for better understanding.
  # for pos in range(len(p_loc)):
  #   canvas[pos[0], pos[1]] = 1

  return canvas.T

def apply_mask(image, maskImage):  
  image = image / image.max()
  image = np.dstack((image, maskImage,image)) 
  return image


def CreateImage(path):
    return  np.array(Image.open(path))

def CreateMask(index,details,mask_data,ImageShape):   
  p_loc = get_pixel_loc(mask_data.loc[index, 'segmentation'], ImageShape)
  maskImage=get_mask(p_loc,ImageShape)
  return maskImage

def CorrectImagesDimensions(width,height,image,maskImage):
    added = height-width
    added2Mask=30
    image = np.concatenate((np.zeros((added,height)),image),axis=0)
    maskImage = np.concatenate((np.zeros((added2Mask,height)),maskImage),axis=0)   
    maskImage = np.concatenate((maskImage,np.zeros((added-added2Mask,height))),axis=0)   
    return image,maskImage


def ShowImages(image,maskImage):
  fig, ax = plt.subplots(1,3, figsize=(10,10))
  ax[0].set_title('Image ('+str(image.shape[0])+','+str(image.shape[1])+')')
  ax[0].imshow(image)
  ax[1].set_title('Mask')    
  ax[1].imshow(maskImage)
  ax[2].set_title('Merged')
  ax[2].imshow(apply_mask(image, maskImage))
  plt.show()

def PrepareImageAndMask(index,mask_data,image_details):
  
  curr_id = mask_data.loc[index, 'id']
  splits = curr_id.split('_')
  x = image_details[(image_details['Case_no']==int(splits[0][4:]))
                  &(image_details['Day']==int(splits[1][3:]))
                  &(image_details['Slice_no']==int(splits[3]))]


  image=CreateImage(path=x['Path'].values[0])  
  maskImage=CreateMask(index,x,mask_data,image.shape)

  if x.Height.values[0]!=x.Width.values[0]:    
    image,maskImage=CorrectImagesDimensions(x.Width.values[0], x.Height.values[0],image,maskImage)
  return image,maskImage

for i in range(1):
  index = index_list[np.random.randint(0,len(index_list) - 1)]
  image,maskImage=PrepareImageAndMask(index,mask_data,image_details)

  ShowImages(image,maskImage)
  
plt.show()