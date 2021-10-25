from collections import OrderedDict

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

import pandas as pd
from tqdm import tqdm


class APTOSDataset(Dataset):
    def __init__(self, csv_path: str, img_column: str, label_names: list):
        self.data = pd.read_csv(csv_path)
        self.img_column = img_column
        self.label_names = label_names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data.iloc[index]
        image = read_image(data[self.img_column]).float()
        label = data[self.label_names].array
        label = label.astype('float32')
        label = torch.Tensor(label).float()
        return image, label


class APTOSModel:
    def __init__(self, csv_path: str, img_column: str, label_names: list):
        self.dataset = APTOSDataset(csv_path, img_column, label_names)

    def fit(self, epochs=5, batch_size=16):
        dataLoader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        after_module = AfterModule()
        after_loss = AfterLoss()
        for epoch in range(epochs):
            print(f'{epoch + 1}/{epochs}')
            for images, labels in tqdm(dataLoader):
                y = after_module(images)
                loss = after_loss(y, labels)
                print(loss)


class AfterModule(nn.Module):
    def __init__(self):
        super(AfterModule, self).__init__()
        # self.layers = nn.Sequential(
        #         nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=0),
        #         nn.ReLU(),
        #         nn.MaxPool2d(2, 2, padding=0),
        #         nn.Conv2d(10, 16, 3, stride=1, padding=0),
        #         nn.ReLU(),
        #         nn.Conv2d(16, 5, 3, stride=1, padding=0),
        #         nn.ReLU())

        self.con1 = nn.Conv2d(3, 5, kernel_size=3, stride=1, padding=0)

        self.classify = nn.Sequential(
                nn.Flatten(),
                nn.Linear(3 * 3 * 32, 7))

    def forward(self, X):
        X = self.con1(X)
        X = self.flat(X)
        return X


class AfterLoss(nn.Module):
    def __init__(self):
        super(AfterLoss, self).__init__()

    def forward(self, input, target):
        print(input.shape, target.shape)

        return nn.MSELoss(input, target)

import tensorflow as tf
if __name__ == '__main__':
    # train_dataset = APTOSDataset('data/train_data_after.csv', 'ImgPath',
    #                              ['preCST', 'preIRF', 'preSRF', 'prePED', 'preHRF'])
    model = APTOSModel('data/train_data_after.csv', 'ImgPath', ['preCST', 'preIRF', 'preSRF', 'prePED', 'preHRF'])
    model.fit()