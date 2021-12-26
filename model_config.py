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

        self.model_log_path = model_save_path

        if training:
            os.makedirs(model_save_path, exist_ok=True)

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.prediction_name = 'pred'
        self.gpu = gpu
        self.training = training
        self.features = []

        self.index_column = index_column
        self.label_regression = label_regression
        self.label_classify = label_classify

        if gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def set_features(self, features: list):
        self.features = features.copy()

        if self.label_regression in self.features:
            self.features.remove(self.label_regression)

        for label_classify in self.label_classify:
            if label_classify in self.features:
                self.features.remove(label_classify)


class ImageConfig(BaseConfig):
    def __init__(self,
                 name: str = 'Model-Image',
                 epochs=100,
                 batch_size=100,
                 learning_rate=0.01,
                 training=True,
                 gpu=True,
                 model_load_path=None,
                 ):
        super(ImageConfig, self).__init__(
            name,
            epochs,
            batch_size,
            learning_rate,
            training,
            gpu,
            model_load_path,
            index_column='patient ID',
            label_regression='CST',
            label_classify=['IRF', 'SRF', 'PED', 'HRF'], )

        self.img_column = 'processed_path'


class CSVConfig(BaseConfig):
    def __init__(self,
                 name: str = 'Model-CSV',
                 epochs=100,
                 batch_size=256,
                 learning_rate=0.01,
                 training=True,
                 gpu=True,
                 model_load_path=None,
                 ):
        super(CSVConfig, self).__init__(
            name,
            epochs,
            batch_size,
            learning_rate,
            training,
            gpu,
            model_load_path,
            index_column='patient ID',
            label_regression='VA',
            label_classify=['continue injection'])

        self.set_features(['gender', 'age', 'diagnosis', 'anti-VEGF', 'L0R1', 'preVA', 'preCST',
                           'preIRF', 'preSRF', 'prePED', 'preHRF', 'CST', 'IRF', 'SRF', 'PED', 'HRF'])
