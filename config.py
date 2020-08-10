"""
Config globals. Should be refactored later...
"""
import os
orig_img_shape = (256, 1600)
img_size = (256, 1600)


#число классов дефектов
num_classes = 4

#параметры обучения
batch_size = 5
epochs = 80

#source dataset and images path
folder_path = os.environ["DATASETS"] + "/steel/"

#image resize coefficient
resize_coeff = 2