import cv2
import numpy as np

from utils import show_func as sf

from random import choice, uniform
import os


def noise_foreground_generate(ori, mask, texture, trans_range=(0.15, 1)):
    texture_image_path = choice(os.listdir(texture))
    random_num = uniform(1, 10)
    if random_num <= 5:
        texture_image = texture_image_path
        i_n = cv2.imread(texture + texture_image)
        i_n = cv2.resize(i_n, ori.shape[:2])
    else:
        i_n = ori

    factor = uniform(trans_range[0], trans_range[1])
    i_n_r_1, i_n_r_2 = cv2.bitwise_and(i_n, i_n, mask=mask), cv2.bitwise_and(ori, ori, mask=mask)
    i_n_r = factor * i_n_r_1 + (1 - factor) * i_n_r_2
    return i_n_r
