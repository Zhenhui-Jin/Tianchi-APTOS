import os
import time

import torch

import config


class BaseConfig:
    """
    基础模型参数
    """

    def __init__(self,
                 name: str,
                 epochs: int,
                 batch_size: int,
                 learning_rate,
                 training,
                 gpu,
                 model_load_path,
                 index_column,
                 img_column,
                 label_regression,
                 label_classify,
                 ):
        self.model_name = name

        time_strftime = time.strftime("%Y%m%d%H%M")
        model_save_path = os.path.join(config.MODEL_FILE_PATH, self.model_name, time_strftime)
        self.model_save_path = os.path.join(model_save_path, f'{self.model_name}.pt')

        if model_load_path:
            self.model_load_path = model_load_path
        else:
            self.model_load_path = self.model_save_path

        # self.model_log_path = os.path.join(model_save_path, 'log')
        # os.makedirs(self.model_log_path, exist_ok=True)
        self.model_log_path = model_save_path

        result_path = os.path.join(config.PREDICT_RESULT_PATH, self.model_name)

        if training:
            os.makedirs(model_save_path, exist_ok=True)
        else:
            os.makedirs(result_path, exist_ok=True)

        result_file_name = f'submission_({self.model_name})_{time.strftime("%Y%m%d")}.csv'
        self.model_result_path = os.path.join(result_path, result_file_name)

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.prediction_name = 'pred'
        self.gpu = gpu
        self.training = training
        self.features = []

        self.index_column = index_column
        self.img_column = img_column
        self.label_regression = label_regression
        self.label_classify = label_classify

    def set_features(self, features: list, use_label=True):
        self.features = features.copy()
        # for no_train in no_train_features:
        #     if no_train in self.features:
        #         self.features.remove(no_train)
        # if not use_label and self.target in self.features:
        #     self.features.remove(self.target)


class ImageConfig(BaseConfig):
    def __init__(self,
                 name: str = 'ImageModel',
                 epochs=100,
                 batch_size=100,
                 learning_rate=0.01,
                 training=True,
                 gpu=True,
                 model_load_path=None,
                 index_column='patient ID',
                 img_column='processed_path',
                 label_regression='CST',
                 label_classify=['IRF', 'SRF', 'PED', 'HRF'],
                 ):
        super(ImageConfig, self).__init__(
            name,
            epochs,
            batch_size,
            learning_rate,
            training,
            gpu,
            model_load_path,
            index_column,
            img_column,
            label_regression,
            label_classify)

        if gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
