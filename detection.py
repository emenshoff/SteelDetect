"""
Steel defects detection
"""

from segmentation_models import Unet
from dataset import train, img_size, height, width, input_shape
from config import folder_path
from utils import rle2maskResize, mask2pad, mask2contour


from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing

from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from PIL import Image
import numpy as np
import cv2

from datetime import date

from tensorflow.keras import utils

import warnings
warnings.filterwarnings("ignore")


#loss function for Unet
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# Keras Sequence-based generator
class DataGenerator(Sequence):
    def __init__(self,
                 df,
                 batch_size=16,
                 subset="train",
                 shuffle=False,
                 preprocess=None, #v-flip and h-flip should be added later...
                 info={}):
        super().__init__()
        self.df = df
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.info = info

        if self.subset == "train":
            self.data_path = folder_path + 'train_images/'
        elif self.subset == "test":
            self.data_path = folder_path + 'test_images/'
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        X = np.empty((self.batch_size, height, width, 3), dtype=np.float32)
        y = np.empty((self.batch_size, height, width, 4), dtype=np.int8)
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        for i, f in enumerate(self.df['ImageId'].iloc[indexes]):
            self.info[index * self.batch_size + i] = f
            X[i,] = Image.open(self.data_path + f).resize(img_size[:-1])
            if self.subset == 'train':
                for j in range(4):
                    y[i, :, :, j] = rle2maskResize(self.df['e' + str(j + 1)].iloc[indexes[i]])
        if self.preprocess != None:
            X = self.preprocess(X)
        if self.subset == 'train':
            return X, y
        else:
            return X


class Detection(object):
#Unet-based classifier and segmenter
    def __init__(self, input_shape=input_shape, num_classes=4, load_weights_from=None):
        self._fitted = False
        self.history = None
        self.model = Unet('resnet34', input_shape=input_shape, classes=num_classes, activation='sigmoid')
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])
        if load_weights_from:
            self.model.load_weights(load_weights_from)
            self._fitted = True


    def fit_model(self, df=train, epochs=20, fit_callback=None):

        callbacks = [
            ModelCheckpoint('saved_models/detection-{epoch:02d}-{dice_coef:.4f}.hdf5',
                            monitor='dice_coef',
                            verbose=1,
                            save_best_only=True,
                            mode='max')#, EarlyStopping(monitor='dice_coef', patience=3)
                    ]

        if fit_callback:
            callbacks.append(fit_callback)

        idx = int(0.8 * len(df))
        preprocess = get_preprocessing('resnet34')
        train_batches = DataGenerator(df.iloc[:idx], shuffle=True, preprocess=preprocess)
        valid_batches = DataGenerator(df.iloc[idx:], preprocess=preprocess)
        self.history = self.model.fit_generator(train_batches,
                                                validation_data=valid_batches,
                                                callbacks = callbacks,
                                                epochs=epochs,
                                                verbose=2)
        self._fitted = True

        #self.model.save_weights(folder_path + "model_weigths.dat")


    def process_image(self, img):

        """gets image stored in numpy array, processes it and retruns tuple (segmented_image, description)"""

        img = cv2.resize(img, img_size[::-1], interpolation=cv2.INTER_AREA)
        img2pred = np.expand_dims(img, axis=0)

        pred = self.model.predict(img2pred)
        pred = np.argmax(pred, axis=3)

        Yclass_answ = np.zeros(img_size)

        for y in range(img_size[0]):
            for x in range(img_size[1]):
                Yclass_answ[y, x] = pred[0, y, x]

        Yresult = np.zeros(input_shape)

        found_defects = []

        for y in range(img_size[0]):
            for x in range(img_size[1]):
                cl = Yclass_answ[y, x]
                # defect class 0 yellow
                if cl == 0:
                    found_defects.append(1)
                    Yresult[y, x, 0] = 255
                    Yresult[y, x, 1] = 255
                    Yresult[y, x, 2] = 0
                # defect class 1 green
                if cl == 1:
                    found_defects.append(2)
                    Yresult[y, x, 0] = 0
                    Yresult[y, x, 1] = 255
                    Yresult[y, x, 2] = 0
                # defect class 3 blue
                if cl == 2:
                    found_defects.append(3)
                    Yresult[y, x, 0] = 0
                    Yresult[y, x, 1] = 0
                    Yresult[y, x, 2] = 255
                # defect class 4 red
                if cl == 3:
                    found_defects.append(4)
                    Yresult[y, x, 0] = 255
                    Yresult[y, x, 1] = 0
                    Yresult[y, x, 2] = 0

        description = ""

        if len(found_defects) == 0:
            description = "No defects found..."
        else:
            description = "Found defects: \n"
            if 1 in  found_defects:
                description += "Class 1: yellow regions\n"
            if 2 in found_defects:
                description += "Class 2: green regions\n"
            if 3 in found_defects:
                description += "Class 3: blue regions\n"
            if 4 in found_defects:
                description += "Class 4: red regions\n"

        return (Yresult, description)



