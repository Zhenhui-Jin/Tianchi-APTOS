import os

import pandas as pd
from pandas import DataFrame

import config


class DataLoad:
    def __init__(self, train_path: str, test_path: str):
        self.__train_images__ = {}  # {'PatientID':[ImgPaths]}
        self.__test_images__ = {}  # {'PatientID':[ImgPaths]}

        self.__load_all_image_path__(train_path, self.__train_images__)
        self.__load_all_image_path__(test_path, self.__test_images__)

        self.train_data: DataFrame = self.__load_csv__(train_path)
        self.test_data: DataFrame = self.__load_csv__(test_path)

    def get_images_train(self, patient_id: str) -> list:
        """
        根据PatientID获取图像列表
        :param patient_id: PatientID
        :return: list
        """
        return self.__train_images__.get(patient_id)

    def get_images_test(self, patient_id: str) -> list:
        return self.__test_images__.get(patient_id)

    @staticmethod
    def __load_csv__(path) -> DataFrame:
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

    def __load_all_image_path__(self, path: str, images: dict):
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
                file_id = file.split('_')[0]
                paths = []
                if images.get(file_id):
                    paths = images[file_id]
                paths.append(file_path)
                images[file_id] = paths


dataLoad = DataLoad(config.TRAIN_DATA_FILE, config.TEST_DATA_FILE)

if __name__ == '__main__':
    print(dataLoad.get_images_train('0000-0501L'))
    print(dataLoad.train_data.info())
