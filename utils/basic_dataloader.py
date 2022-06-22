from torch.utils.data import dataset, DataLoader
from torchvision import transforms
from anomaly_simulation.step_1 import *
from anomaly_simulation.step_2 import *
from anomaly_simulation.step_3 import *

import os

from utils import show_func as sf


class MemsegDataset(dataset.Dataset):
    def __init__(self, image_directory, gt_directory, texture_directory, train_mode=True):
        super(MemsegDataset, self).__init__()
        self.image_directory = image_directory
        self.gt_directory = gt_directory
        self.texture_directory = texture_directory
        self.mode = train_mode

        self.transforms = transforms.Compose(
            [transforms.Resize(256),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5], std=[0.5])
             ]
        )

    def __getitem__(self, index):
        assert len(os.listdir(self.image_directory)) == len(os.listdir(self.gt_directory))
        image_list, gt_list = os.listdir(self.image_directory), os.listdir(self.gt_directory)
        img_path, gt_path = image_list[index], gt_list[index]

        original_image = cv2.imread(self.image_directory + img_path)
        mask_image = MaskImage(path=self.image_directory + img_path).process()
        noisy_image = noise_foreground_generate(ori=original_image, mask=mask_image, texture=self.texture_directory)
        image = simulated_generate(mask=mask_image, ori=original_image, noisy=noisy_image)

        gt = cv2.imread(self.gt_directory + gt_path)

        return image, gt

    def __len__(self):
        return len(os.listdir(self.image_directory))


def BaseDataloader(iD, gD, tD, train_mode=True, batch_size=16):
    the_dataset = MemsegDataset(image_directory=iD, gt_directory=gD, texture_directory=tD, train_mode=train_mode)
    loader = DataLoader(the_dataset, batch_size=batch_size, shuffle=True)
    return loader


if __name__ == '__main__':
    iD = r"G:/Dataset/mvtec_anomaly_detection/mvtec_anomaly_detection/carpet/test/color/"
    gD = r'G:/Dataset/mvtec_anomaly_detection/mvtec_anomaly_detection/carpet/ground_truth/color/'
    tD = r'G:/Dataset/DTD/dtd/images/banded/'

    the_dataloader = BaseDataloader(iD, gD, tD, train_mode=True)

    for idx, (image, gt) in enumerate(the_dataloader):
        print(gt.shape)