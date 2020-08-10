"""
Подготовка датасета
"""

import pandas as pd
import numpy as np
import os
import cv2
import pandas as pd
from utils import rle2maskResize, mask2pad, mask2contour
from config import folder_path, resize_coeff,  orig_img_shape



df = pd.read_csv(folder_path + 'train.csv')

# Перетряхиваем исходный датафрейм
df['ImageId'] = df['ImageId_ClassId'].map(lambda x: x.split('.')[0]+'.jpg')

# Убираем картинки без размеченных дефектов
#df = df[df['EncodedPixels'].notnull()]

train = pd.DataFrame({'ImageId': df['ImageId'][::4]})

train['e1'] = df['EncodedPixels'][::4].values
train['e2'] = df['EncodedPixels'][1::4].values
train['e3'] = df['EncodedPixels'][2::4].values
train['e4'] = df['EncodedPixels'][3::4].values

train.reset_index(inplace=True, drop=True)

train.fillna('', inplace=True)

train['count'] = np.sum(train.iloc[:, 1:] != '', axis=1).values

#Получение списка файлов из датасета
def get_img_list(df):
    fnames = df['ImageId'].unique()
    fnames = fnames.tolist()
    return fnames

images_list = get_img_list(train)
images_path = folder_path + "train_images/"
images_list = os.listdir(images_path)

#берем размер картинки
sample_image = images_list[0]
img = cv2.imread(images_path + f'{sample_image}' )
orig_img_shape = img.shape


#размеры тензора на входе сети
input_shape = (orig_img_shape[0] // resize_coeff,
               orig_img_shape[1] // resize_coeff,
               orig_img_shape[2])

#размер картинки
img_size = input_shape[0:2]

height, width = img_size








