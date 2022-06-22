import cv2
import numpy as np

from utils import show_func as sf


def simulated_generate(mask, ori, noisy):
    mask = np.where(mask == 255, 0, 255)
    i_r = cv2.bitwise_and(ori, ori, mask=mask)
    sf.cv_show(i_r, 'i_r')
    res = i_r + noisy
    return res
