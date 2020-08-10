# функции работы с масками. Код взят с источника:
# https://www.kaggle.com/titericz/building-and-visualizing-masks

import numpy as np
import pandas as pd
from config import orig_img_shape, img_size


def rle2maskResize(rle):
    # CONVERT RLE TO MASK
    if (pd.isnull(rle)) | (rle == ''):
        return np.zeros(img_size,
                        dtype=np.uint8)
    width = orig_img_shape[1]
    height = orig_img_shape[0]
    mask = np.zeros(width * height, dtype=np.uint8)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2] - 1
    lengths = array[1::2]
    for index, start in enumerate(starts):
        mask[int(start): int(start + lengths[index])] = 1

    return mask.reshape((height, width), order='F')[::2, ::2]


def mask2contour(mask, width=3):
    # CONVERT MASK TO ITS CONTOUR
    w = mask.shape[1]
    h = mask.shape[0]
    mask2 = np.concatenate([mask[:, width:], np.zeros((h, width))], axis=1)
    mask2 = np.logical_xor(mask, mask2)
    mask3 = np.concatenate([mask[width:, :], np.zeros((width, w))], axis=0)
    mask3 = np.logical_xor(mask, mask3)
    return np.logical_or(mask2, mask3)


def mask2pad(mask, pad=2):
    # ENLARGE MASK TO INCLUDE MORE SPACE AROUND DEFECT
    w = mask.shape[1]
    h = mask.shape[0]

    # MASK UP
    for k in range(1, pad, 2):
        temp = np.concatenate([mask[k:, :], np.zeros((k, w))], axis=0)
        mask = np.logical_or(mask, temp)
    # MASK DOWN
    for k in range(1, pad, 2):
        temp = np.concatenate([np.zeros((k, w)), mask[:-k, :]], axis=0)
        mask = np.logical_or(mask, temp)
    # MASK LEFT
    for k in range(1, pad, 2):
        temp = np.concatenate([mask[:, k:], np.zeros((h, k))], axis=1)
        mask = np.logical_or(mask, temp)
    # MASK RIGHT
    for k in range(1, pad, 2):
        temp = np.concatenate([np.zeros((h, k)), mask[:, :-k]], axis=1)
        mask = np.logical_or(mask, temp)

    return mask

