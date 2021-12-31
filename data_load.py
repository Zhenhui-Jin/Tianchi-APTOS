import os

import cv2
import numpy as np
import pandas as pd
from pandas import DataFrame

import config

np.random.seed(1)


class DataLoad:
    def __init__(self, train_path: str, test_path: str):
        self.train_path = train_path
        self.test_path = test_path

        train_images = []  # [{'patient ID': id, 'ImgNumber': number, 'ImgPath': path}]
        test_images = []  # [{'patient ID': id, 'ImgNumber': number, 'ImgPath': path}]
        self.__load_all_image_path__(train_path, train_images)
        self.__load_all_image_path__(test_path, test_images)

        self.train_images = pd.DataFrame(train_images, columns=['patient ID', 'LR', 'ImgNumber', 'ImgPath'])
        self.test_images = pd.DataFrame(test_images, columns=['patient ID', 'LR', 'ImgNumber', 'ImgPath'])

        self.train_data: DataFrame = pd.merge(self.get_train_csv(), self.train_images, on='patient ID')
        self.test_data: DataFrame = pd.merge(self.get_test_csv(), self.test_images, on='patient ID')

    def get_train_csv(self):
        return self.load_csv(self.train_path)

    def get_test_csv(self):
        return self.load_csv(self.test_path)

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
                data = pd.read_csv(os.path.join(path, file), header=0)
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
                if len(split) > 2:
                    file_number = split[1] + split[2][-3:]
                else:
                    file_number = split[1]
                # _1_000000
                images.append({'patient ID': file_id,
                               'LR': file_id[-1],
                               'ImgNumber': file_number,
                               'ImgPath': file_path.replace(config.root_path + '\\', '').replace('\\', '/')})

    @staticmethod
    def read_image(path: str, to_float=False):
        """
        使用OpenCV加载图像数据
        :param path:
        :param to_float:
        :return:
        """
        img: np.ndarray = cv2.imread(path)
        # 裁剪
        # img = img[:500, 500:, :]
        if to_float:
            img = img.astype('float32')
            img = img / 255.0
        return img

    def get_train_data_cst_all(self, size=None, read_img=False, to_float=False) -> pd.DataFrame:
        """
        CST，获取治疗前和治疗后的训练图像，各取size个样本
        :param size:
        :param read_img:
        :return:
        """
        pre_cst = self.get_train_data_pre_cst(size, read_img, to_float)
        cst = self.get_train_data_cst(size, read_img, to_float)
        pre_cst = pre_cst.rename({'preCST': 'label', 'Img': 'feature', 'ImgPath': 'feature'}, axis=1)
        pre_cst['type'] = 'preCST'
        cst = cst.rename({'CST': 'label', 'Img': 'feature', 'ImgPath': 'feature'}, axis=1)
        cst['type'] = 'CST'
        cst_data = pd.concat([pre_cst, cst], axis=0, ignore_index=True)
        return cst_data[['patient ID', 'type', 'label', 'feature']]

    def get_train_data_pre_cst(self, size=None, read_img=False, to_float=False) -> pd.DataFrame:
        """
        获取治疗前的训练数据 preCST
        :param size: 获取样本数量，None为获取全部
        :param read_img 是否读取图像，False：不读取，返回'ImgPath'；True：读取，返回'Img'
        :return: ['patient ID', 'preCST', 'ImgPath'|'Img']
        """

        train_data = self.train_data[self.train_data['ImgNumber'].apply(lambda x: x.startswith('10'))]
        img_number = train_data.groupby(['patient ID']).count()['ImgNumber'] / 2 + 1000
        img_number = img_number.astype(int)

        train_data = train_data[
            train_data.apply(lambda x: img_number[x.loc['patient ID']] == int(x.loc['ImgNumber']), axis=1)]
        # train_data.loc[train_data['preCST'].isna(), 'preCST'] = 0
        train_data.dropna(subset=['preCST'], inplace=True, axis=0)

        if size is not None:
            train_data = train_data.head(size)

        if read_img:
            train_data['Img'] = train_data['ImgPath'].apply(lambda path: np.array(self.read_image(path, to_float)))
            return train_data[['patient ID', 'preCST', 'Img']]
        else:
            return train_data[['patient ID', 'preCST', 'ImgPath']]

    def get_train_data_cst(self, size=None, read_img=False, to_float=False) -> pd.DataFrame:
        """
        获取治疗后的训练数据 CST
        :param size: 获取样本数量，None为获取全部
        :param read_img 是否读取图像，False：不读取，返回'ImgPath'；True：读取，返回'Img'
        :return: ['patient ID', 'CST', 'ImgPath'|'Img']
        """

        train_data = self.train_data[self.train_data['ImgNumber'].apply(lambda x: x.startswith('20'))]
        img_number = train_data.groupby(['patient ID']).count()['ImgNumber'] / 2 + 2000
        img_number = img_number.astype(int)

        train_data = train_data[
            train_data.apply(lambda x: img_number[x.loc['patient ID']] == int(x.loc['ImgNumber']), axis=1)]
        # train_data.loc[train_data['CST'].isna(), 'preCST'] = 0
        train_data.dropna(subset=['CST'], inplace=True, axis=0)
        if size is not None:
            train_data = train_data.head(size)

        if read_img:
            train_data['Img'] = train_data['ImgPath'].apply(lambda path: np.array(self.read_image(path, to_float)))
            return train_data[['patient ID', 'CST', 'Img']]
        else:
            return train_data[['patient ID', 'CST', 'ImgPath']]

    def get_test_data_cst_all(self, size=None, read_img=False, to_float=False) -> pd.DataFrame:
        """
        CST，获取治疗前和治疗后的预测图像，各取size个样本
        :param size:
        :param read_img:
        :param to_float:
        :return:
        """
        pre_cst = self.get_test_data_pre_cst(size, read_img, to_float)
        cst = self.get_test_data_cst(size, read_img, to_float)
        pre_cst = pre_cst.rename({'Img': 'feature', 'ImgPath': 'feature'}, axis=1)
        pre_cst['type'] = 'preCST'
        cst = cst.rename({'Img': 'feature', 'ImgPath': 'feature'}, axis=1)
        cst['type'] = 'CST'
        cst_data = pd.concat([pre_cst, cst], axis=0, ignore_index=True)
        return cst_data[['patient ID', 'type', 'feature']]

    def get_test_data_pre_cst(self, size=None, read_img=False, to_float=False) -> pd.DataFrame:
        """
        获取治疗前的预测数据 preCST
        :param size: 获取样本数量，None为获取全部
        :param read_img 是否读取图像，False：不读取，返回'ImgPath'；True：读取，返回'Img'
        :return: ['patient ID', 'ImgPath'|'Img']
        """
        test_data = self.test_data[self.test_data['ImgNumber'].apply(lambda x: x.startswith('10'))]
        img_number = test_data.groupby(['patient ID']).count()['ImgNumber'] / 2 + 1000
        img_number = img_number.astype(int)

        test_data = test_data[
            test_data.apply(lambda x: img_number[x.loc['patient ID']] == int(x.loc['ImgNumber']), axis=1)]
        if size is not None:
            test_data = test_data.head(size)

        if read_img:
            test_data['Img'] = test_data['ImgPath'].apply(lambda path: np.array(self.read_image(path, to_float)))
            return test_data[['patient ID', 'Img']]
        else:
            return test_data[['patient ID', 'ImgPath']]

    def get_test_data_cst(self, size=None, read_img=False, to_float=False) -> pd.DataFrame:
        """
        获取治疗前的预测数据 CST
        :param size: 获取样本数量，None为获取全部
        :param read_img 是否读取图像，False：不读取，返回'ImgPath'；True：读取，返回'Img'
        :return: ['patient ID', 'ImgPath'|'Img']
        """
        test_data = self.test_data[self.test_data['ImgNumber'].apply(lambda x: x.startswith('20'))]
        img_number = test_data.groupby(['patient ID']).count()['ImgNumber'] / 2 + 2000
        img_number = img_number.astype(int)

        test_data = test_data[
            test_data.apply(lambda x: img_number[x.loc['patient ID']] == int(x.loc['ImgNumber']), axis=1)]
        if size is not None:
            test_data = test_data.head(size)

        if read_img:
            test_data['Img'] = test_data['ImgPath'].apply(lambda path: np.array(self.read_image(path, to_float)))
            return test_data[['patient ID', 'Img']]
        else:
            return test_data[['patient ID', 'ImgPath']]


if __name__ == '__main__':
    dataLoad = DataLoad(config.TRAIN_DATA_FILE_NEW, config.TEST_DATA_FILE_NEW)

    train_data = dataLoad.get_train_data_cst_all(3, read_img=True)
    print(train_data)
    print(train_data.shape)

    train_data = dataLoad.get_test_data_cst_all(3, read_img=True)
    print(train_data)
    print(train_data.shape)
