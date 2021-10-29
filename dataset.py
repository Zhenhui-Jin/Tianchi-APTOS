from collections import OrderedDict

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

import pandas as pd
from tqdm import tqdm


class APTOSDataset(Dataset):
    def __init__(self, csv_path: str, img_column: str, label_regression: str, label_classify: list):
        """
        数据加载器
        :param csv_path: 整理好的csv数据文件路径
        :param img_column: 图像路径的column
        :param label_regression: 做回归训练的标签名称
        :param label_classify: 做分类训练的标签名称
        """
        self.data = pd.read_csv(csv_path).iloc[:50, :]
        self.img_column = img_column
        self.label_regression = label_regression
        self.label_classify = label_classify

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        返回一条数据：
        image:图像，regression_label:回归的label，classify_label:分类的label
        :param index:
        :return: image, (regression_label, classify_label)
        """
        data = self.data.iloc[index]
        image = read_image(data[self.img_column]).float()
        regression_label = None
        if self.label_regression:
            regression_label = data[[self.label_regression]].array
            regression_label = regression_label.astype('float32')
            regression_label = torch.Tensor(regression_label).float()

        classify_label = None
        if self.label_classify:
            classify_label = data[self.label_classify].array
            classify_label = classify_label.astype('float32')
            classify_label = torch.Tensor(classify_label).float()
        return image, (regression_label, classify_label)


class APTOSModel:
    def __init__(self, csv_path: str, img_column: str, label_regression: str, label_classify: list):
        self.dataset = APTOSDataset(csv_path, img_column, label_regression, label_classify)

    def fit(self, epochs=5, batch_size=16):
        dataLoader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        after_module = AfterModule()
        loss = AfterLoss()
        for epoch in range(epochs):
            print(f'{epoch + 1}/{epochs}')
            for images, (regression_labels, classify_labels) in tqdm(dataLoader):
                y_regressions, y_classifys = after_module(images)
                loss = loss(regression_labels, classify_labels, y_regressions, y_classifys)
                print(loss)


class AfterModule(nn.Module):
    def __init__(self):
        super(AfterModule, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, padding=0),
            nn.Conv2d(10, 16, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, padding=0),
            nn.Conv2d(16, 5, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, padding=0),
            nn.Flatten(),
            nn.ReLU()
        )

        self.regression = nn.Linear(16*3*16*3*10*3, 1)
        self.classify = nn.Linear(918555, 7)

    def forward(self, X):
        X = self.layers(X)
        y_regression = self.regression(X)
        y_classify = self.classify(X)
        return y_regression, y_classify


class AfterLoss(nn.Module):
    def __init__(self):
        super(AfterLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.cross_loss = nn.CrossEntropyLoss()

    def forward(self, input, target):
        print(target)
        print(input.shape, target.shape)

        mse_loss = nn.MSELoss(input, target)
        cross_entropy_loss = nn.CrossEntropyLoss(input, target)
        return mse_loss * 1 + cross_entropy_loss * 0.5


if __name__ == '__main__':
    # train_dataset = APTOSDataset('data/train_data_after.csv', 'ImgPath',
    #                              ['preCST', 'preIRF', 'preSRF', 'prePED', 'preHRF'])
    model = APTOSModel('data/train_data_after.csv', 'ImgPath', 'preCST', ['preIRF', 'preSRF', 'prePED', 'preHRF'])
    model.fit()
