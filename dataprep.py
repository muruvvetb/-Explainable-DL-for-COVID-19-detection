import torch
import os
import torch.utils.data
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2 as cv
from torch.utils.data import DataLoader


def clahe(image):
    lab_image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab_image)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(l)
    updated_lab_image = cv.merge((clahe_img, a, b))
    clahe_img = cv.cvtColor(updated_lab_image, cv.COLOR_LAB2RGB)
    return Image.fromarray(clahe_img)


class DataReader(torch.utils.data.Dataset):
    def __init__(self, mode='train', dataset_path=None, csv_path=None, image_size=224):
        super(DataReader, self).__init__()

        self.target_classes = []
        self.input_img_paths = []

        if mode == 'train':
            self.input_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomAffine(5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
            ])

        else:
            self.input_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
            ])

        data = np.loadtxt(csv_path + "/{}.csv".format(mode), delimiter=',', dtype=str)

        # classes["Normal", "COVID-19", "Virus", "bacteria"]

        self.input_img_paths = [os.path.join(os.path.join(dataset_path, item[1]), item[0]) for item in data]
        classes = lambda x: {"Normal": 0, "COVID-19": 1, "Virus": 2, "bacteria": 3}[x]
        self.target_classes = list(map(classes, data[:, 1]))

    def load_input_img(self, filepath):
        img = cv.imread(filepath)
        return img

    def __getitem__(self, index):

        input_img = self.load_input_img(self.input_img_paths[index])
        input_img = clahe(input_img)
        input_img = self.input_transform(input_img)
        return input_img, self.target_classes[index]

    def __len__(self):

        return len(self.input_img_paths)


def get_dataloader(batch_size, dataset_path=None, csv_path=None, image_size=224, device='cpu'):
    kwargs = {"num_workers": 4, "pin_memory": True} if device!='cpu' else {}
    train_data = DataReader(mode="train", dataset_path=dataset_path, csv_path=csv_path, image_size=image_size)
    val_data = DataReader(mode="val", dataset_path=dataset_path, csv_path=csv_path, image_size=image_size)
    test_data = DataReader(mode="test", dataset_path=dataset_path, csv_path=csv_path, image_size=image_size)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  **kwargs)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,  **kwargs)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,  **kwargs)
    return train_loader, val_loader, test_loader