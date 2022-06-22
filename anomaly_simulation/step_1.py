import cv2
from noise import pnoise2
import numpy as np
from utils import show_func as sf


class MaskImage:
    def __init__(self, path, octaves=2, persistence=0.5, lacunarity=2.0):
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity

        self.path = path
        self.img = cv2.imread(self.path)
        self.shape = self.img.shape[0:2]

    def _binary_mask(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        return binary

    def _noise_generate(self):
        temp = np.zeros(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                temp[i][j] = pnoise2(i / 100, j / 100,
                                     octaves=self.octaves,
                                     persistence=self.persistence,
                                     lacunarity=self.lacunarity,
                                     repeatx=self.shape[0],
                                     repeaty=self.shape[1])
        temp *= 100
        temp = temp.astype(dtype=np.uint8)
        _, temp = cv2.threshold(temp, 120, 255, cv2.THRESH_BINARY)
        return temp

    def _mask_generate(self):
        m_i = self._binary_mask()
        m_p = self._noise_generate()
        m_m = cv2.bitwise_and(m_i, m_p)
        return m_m

    def process(self):
        return self._mask_generate()
