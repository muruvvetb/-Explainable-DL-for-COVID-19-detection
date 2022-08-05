import torch
import os
import torch.utils.data
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2 as cv
from torch.utils.data import DataLoader


class DataReader(torch.utils.data.Dataset):
    def __init__(self, mode='train', dataset_path=None, csv_path=None, image_size=224, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        super(DataReader, self).__init__()

        self.target_classes = []
        self.input_img_paths = []

        if mode == 'train':
            self.input_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,
                                     std=std)
            ])

        else:
            self.input_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,
                                     std=std)
            ])

        data = np.loadtxt(csv_path + "/{}.csv".format(mode), delimiter=',', dtype=str)

        # classes["Normal", "COVID-19", "Virus", "bacteria"]

        self.input_img_paths = [os.path.join(os.path.join(dataset_path, item[1]), item[0]) for item in data]
        classes = lambda x: {"Normal": 0, "COVID-19": 1, "Virus": 2, "bacteria": 3}[x]
        self.target_classes = list(map(classes, data[:, 1]))

    def load_input_img(self, filepath):
        rgb_img = cv.imread(filepath, 1)[:, :, ::-1]
        rgb_img = cv.resize(rgb_img, (224, 224), interpolation=cv.INTER_AREA)
        rgb_img = np.float32(rgb_img) / 255

        return  rgb_img

    def __getitem__(self, index):
        input_img = self.load_input_img(self.input_img_paths[index])
        input_img = self.input_transform(input_img)
        return input_img, self.target_classes[index]

    def __len__(self):

        return len(self.input_img_paths)


def get_dataloader(batch_size, dataset_path=None, csv_path=None, image_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device='cpu'):
    kwargs = {"num_workers": 4, "pin_memory": True} if device!='cpu' else {}
    train_data = DataReader(mode="train", dataset_path=dataset_path, csv_path=csv_path, image_size=image_size, mean = mean, std = std)
    val_data = DataReader(mode="val", dataset_path=dataset_path, csv_path=csv_path, image_size=image_size, mean = mean, std = std)
    test_data = DataReader(mode="test", dataset_path=dataset_path, csv_path=csv_path, image_size=image_size, mean = mean, std = std)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  **kwargs)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,  **kwargs)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,  **kwargs)
    return train_loader, val_loader, test_loader
