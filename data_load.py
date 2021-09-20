import os

import cv2

import numpy as np
import pandas as pd
from pandas import DataFrame

import config

np.random.seed(1)


class DataLoad:
    def __init__(self, train_path: str, test_path: str):
        train_images = []  # [{'patient ID': id, 'ImgNumber': number, 'ImgPath': path}]
        test_images = []  # [{'patient ID': id, 'ImgNumber': number, 'ImgPath': path}]
        self.__load_all_image_path__(train_path, train_images)
        self.__load_all_image_path__(test_path, test_images)

        self.train_images = pd.DataFrame(train_images)
        self.test_images = pd.DataFrame(test_images)

        self.train_csv: DataFrame = self.load_csv(train_path)
        self.test_csv: DataFrame = self.load_csv(test_path)

        self.train_data: DataFrame = None
        self.test_data: DataFrame = None

        self.merge_csv_image()

    def merge_csv_image(self):
        """
        将CSV数据和对应图像路径合并
        :return:
        """
        self.train_data: DataFrame = pd.merge(self.train_csv, pd.DataFrame(self.train_images), on='patient ID')
        self.test_data: DataFrame = pd.merge(self.test_csv, pd.DataFrame(self.test_images), on='patient ID')

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
    def read_image(path: str):
        """
        使用OpenCV加载图像数据
        :param path:
        :return:
        """
        img: np.ndarray = cv2.imread(path)
        # 裁剪
        img = img[:500, 500:, :]
        img = img.astype('float32')
        img = img / 255
        return img

    def get_pre_cst_train_data(self, size=None) -> (np.ndarray, np.ndarray):
        """
        获取治疗前的训练数据 preCST
        :param size: 获取样本数量，None为获取全部
        :return: data, label
        """
        if self.train_data is None:
            raise Exception("先调用 merge_csv_image()")

        def numberF(x):
            return imgNumber[x.loc['patient ID']] == int(x.loc['ImgNumber'])

        train_data = self.train_data[self.train_data['ImgNumber'].apply(lambda x: x.startswith('10'))]
        imgNumber = train_data.groupby(['patient ID']).count()['ImgNumber'] / 2 + 1000
        imgNumber = imgNumber.astype(int)
        train_data = train_data[train_data.apply(numberF, axis=1)]
        # train_data.loc[train_data['preCST'].isna(), 'preCST'] = train_data['preCST'].mean()
        train_data.dropna(subset=['preCST'], inplace=True, axis=0)
        if size is not None:
            train_data = train_data.head(size)

        data = np.array([self.read_image(path) for path in train_data['ImgPath']])
        label = train_data['preCST'].values

        return data, label

    def get_pre_cst_test_data(self, size=None) -> (np.ndarray, np.ndarray):
        """
        获取治疗前的预测数据 preCST
        :param size: 获取样本数量，None为获取全部
        :return: data, label
        """
        if self.test_data is None:
            raise Exception("先调用 merge_csv_image()")

        def numberF(x):
            return imgNumber[x.loc['patient ID']] == int(x.loc['ImgNumber'])

        test_data = self.test_data[self.test_data['ImgNumber'].apply(lambda x: x.startswith('10'))]
        imgNumber = test_data.groupby(['patient ID']).count()['ImgNumber'] / 2 + 1000
        imgNumber = imgNumber.astype(int)
        test_data = test_data[test_data.apply(numberF, axis=1)]
        test_data.loc[test_data['preCST'].isna(), 'preCST'] = test_data['preCST'].mean()
        if size is not None:
            test_data = test_data.head(size)

        data = np.array([self.read_image(path) for path in test_data['ImgPath']])
        label = test_data['preCST'].values

        return data, label

    def get_cst_test_data(self, size=None):
        """
        获取治疗后的预测数据  CST
        :param size: 获取样本数量，None为获取全部
        :return: data, label
        """
        if self.test_data is None:
            raise Exception("先调用 merge_csv_image()")

        def numberF(x):
            return imgNumber[x.loc['patient ID']] == int(x.loc['ImgNumber'])

        test_data = self.test_data[self.test_data['ImgNumber'].apply(lambda x: x.startswith('20'))]
        imgNumber = test_data.groupby(['patient ID']).count()['ImgNumber'] / 2 + 2000
        imgNumber = imgNumber.astype(int)
        test_data = test_data[test_data.apply(numberF, axis=1)]
        test_data.loc[test_data['preCST'].isna(), 'preCST'] = test_data['preCST'].mean()
        if size is not None:
            test_data = test_data.head(size)

        test_data['Img'] = test_data.apply(lambda x: self.read_image(x['ImgPath']), axis=1)

        data = test_data['Img']
        label = test_data['preCST']

        return data, label

    def get_cst_train_data(self, size=None):
        """
        获取治疗后的训练数据 CST
        :param size: 获取样本数量，None为获取全部
        :return: data, label
        """
        if self.train_data is None:
            raise Exception("先调用 merge_csv_image()")

        def numberF(x):
            return imgNumber[x.loc['patient ID']] == int(x.loc['ImgNumber'])

        train_data = self.train_data[self.train_data['ImgNumber'].apply(lambda x: x.startswith('20'))]
        imgNumber = train_data.groupby(['patient ID']).count()['ImgNumber'] / 2 + 2000
        imgNumber = imgNumber.astype(int)
        train_data = train_data[train_data.apply(numberF, axis=1)]
        train_data.loc[train_data['preCST'].isna(), 'preCST'] = train_data['preCST'].mean()
        if size is not None:
            train_data = train_data.head(size)

        train_data['Img'] = train_data.apply(lambda x: self.read_image(x['ImgPath']), axis=1)

        data = train_data['Img']
        label = train_data['preCST']

        return data, label


dataLoad = DataLoad(config.TRAIN_DATA_FILE, config.TEST_DATA_FILE)

if __name__ == '__main__':
    data, label, = dataLoad.get_pre_cst_train_data(3)
    print(data.shape)
    print(label)
    print(label.dtype)
