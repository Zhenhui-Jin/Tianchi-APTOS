import os

import cv2

import numpy as np
import pandas as pd
from pandas import DataFrame

import config


class DataLoad:
    def __init__(self, train_path: str, test_path: str):
        train_images = []  # [{'patient ID': id, 'ImgNumber': number, 'ImgPath': path}]
        test_images = []  # [{'patient ID': id, 'ImgNumber': number, 'ImgPath': path}]
        self.__load_all_image_path__(train_path, train_images)
        self.__load_all_image_path__(test_path, test_images)

        self.train_images = pd.DataFrame(train_images)
        self.test_images = pd.DataFrame(test_images)

        self.train_data: DataFrame = self.load_csv(train_path)
        self.test_data: DataFrame = self.load_csv(test_path)

    def merge_csv_image(self):
        """
        将CSV数据和对应图像路径合并
        :return:
        """
        self.train_data: DataFrame = pd.merge(self.train_data, pd.DataFrame(self.train_images), on='patient ID')
        self.test_data: DataFrame = pd.merge(self.test_data, pd.DataFrame(self.test_images), on='patient ID')

    @staticmethod
    def load_csv(path) -> DataFrame:
        """
        加载CSV数据集
        :param path: 路径
        :return: DataFrame
        """
        files = os.listdir(path)
        for file in files:
            if file.endswith('.csv'):
                data = pd.read_csv(os.path.join(path, file))
                return data

    def __load_all_image_path__(self, path: str, images: list):
        """
        加载所有图像路径
        :param path:路径
        :param images:
        :return:
        """
        files = os.listdir(path)
        for file in files:
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                self.__load_all_image_path__(file_path, images)
            elif file.endswith('.jpg'):
                split = file.replace('.jpg', '').split('_')
                file_id = split[0]
                file_number = split[1]
                images.append({'patient ID': file_id, 'ImgNumber': file_number, 'ImgPath': file_path})

    @staticmethod
    def read_image(path: str) -> np.ndarray:
        """
        使用OpenCV加载图像数据
        :param path:
        :return:
        """
        img: np.ndarray = cv2.imread(path)
        # 裁剪
        img = img[:500, 500:, :]
        return img


dataLoad = DataLoad(config.TRAIN_DATA_FILE, config.TEST_DATA_FILE)

if __name__ == '__main__':
    print(dataLoad.train_images.head(3))
    print(dataLoad.train_data.info())
    print(dataLoad.train_data.head(3))

    dataLoad.train_data.to_csv('train.csv')
