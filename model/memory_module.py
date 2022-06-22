import os
from collections import defaultdict

import torch.nn as nn
import torch

import cv2

from utils.resnet import resnet18


class MemoryBank:
    def __init__(self, standard_directory):
        self.memory = defaultdict(list)
        self.model = resnet18()
        self.directory = standard_directory
        self._enter_in()

    def _enter_in(self):
        for file in os.listdir(self.directory):
            image = cv2.imread(self.directory + file)
            image = torch.tensor(cv2.resize(image, (256, 256)), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            _, features = self.model(image)
            features = features[1:-1]
            self.memory[file] = features

    def _diff_info_cal(self, x):
        best_one = ['', float('inf')]
        _, x_features = self.model(x)
        x_features = x_features[1:-1]
        for k, v in self.memory.items():
            ans = 0
            for i in range(3):
                loss = nn.MSELoss()
                ans += loss(x_features[i], v[i])
            if ans < best_one[1]:
                best_one = [k, ans]
        return [self.memory[best_one[0]], x_features]

    def _concat_info(self, x):
        concat_feature = []
        for batch in range(x.shape[0]):
            m_feature, x_feature = self._diff_info_cal(x[batch, :, :, :].unsqueeze(0))
            batch_feature = []
            for i in range(3):
                batch_feature.append(torch.cat([m_feature[i], x_feature[i]], dim=1))
            concat_feature.append(batch_feature)
        res = [torch.cat([i[0] for i in concat_feature], dim=0), torch.cat([i[1] for i in concat_feature], dim=0),
               torch.cat([i[2] for i in concat_feature], dim=0)]

        return res

    def process(self, x):
        return self._concat_info(x)
