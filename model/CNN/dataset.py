import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class APTOSDataset(Dataset):
    def __init__(self, csv_path: str, img_column: str, label_regression: str, label_classify: list, training=True):
        """
        数据加载器
        :param csv_path: 整理好的csv数据文件路径
        :param img_column: 图像路径的column
        :param label_regression: 做回归训练的标签名称
        :param label_classify: 做二分类训练的标签名称
        :param training: 是否是训练
        """
        self.data = pd.read_csv(csv_path)

        columns = self.data.columns
        if label_regression in columns:
            self.data.loc[self.data[label_regression].isna(), label_regression] = self.data[label_regression].mean()
        for label in label_classify:
            if label in columns:
                self.data.loc[self.data[label].isna(), label] = 0

        self.img_column = img_column
        self.label_regression = label_regression
        self.label_classify = label_classify
        self.training = training

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        返回一条数据：
        image:图像，regression_label:回归的label，classify_label:分类的label
        :param index:
        :return: image, (regression_label, classify_label); training=True 只有 image
        """
        data = self.data.iloc[index]
        path = data[self.img_column]
        image = read_image(path).float()
        regression_label = None
        columns = self.data.columns.tolist()
        if self.label_regression and self.label_regression in columns:
            regression_label = data[[self.label_regression]].array
            regression_label = torch.Tensor(regression_label).float()

        classify_label = None
        label_names = []
        for name in self.label_classify:
            if name in columns:
                label_names.append(name)

        if self.label_classify and label_names:
            classify_label = data[self.label_classify].array
            classify_label = torch.Tensor(classify_label).float()

        if self.training:
            return image, (regression_label / 100, classify_label)
        else:
            return image
