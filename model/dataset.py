import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from model_config import ImageConfig, CSVConfig


class ImageDataset(Dataset):
    def __init__(self, config: ImageConfig, data: pd.DataFrame):
        """
        数据加载器
        """
        self.data = data.copy()
        self.config = config

        columns = self.data.columns
        if config.label_regression in columns:
            self.data.loc[self.data[config.label_regression].isna(), config.label_regression] = self.data[
                config.label_regression].mean()
        for label in config.label_classify:
            if label in columns:
                self.data.loc[self.data[label].isna(), label] = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        返回一条数据：
        image:图像，regression_label:回归的label，classify_label:分类的label
        :param index:
        :return: training=True: image, (regression_label, classify_label); training=False: image, index_data
        """
        data = self.data.iloc[index]
        image = read_image(data[self.config.img_column]) / 255.0

        regression_label = None
        columns = self.data.columns.tolist()
        if self.config.label_regression and self.config.label_regression in columns:
            regression_label = data[[self.config.label_regression]].array
            regression_label = torch.Tensor(regression_label).float()

        classify_label = None
        label_names = []
        for name in self.config.label_classify:
            if name in columns:
                label_names.append(name)

        if self.config.label_classify and label_names:
            classify_label = data[self.config.label_classify].array
            classify_label = torch.Tensor(classify_label).float()

        if self.config.training:
            return image, (regression_label / 100.0, classify_label)
            # return image, (regression_label, classify_label)
        else:
            index_data = torch.Tensor([index]).int()
            return image, index_data
