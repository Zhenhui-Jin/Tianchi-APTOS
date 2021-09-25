import abc
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from config import MODEL_SAVE_PATH
from data_load import dataLoad


class BaseModel:
    def __init__(self, model_name, input_shape,
                 learning_rate=0.001,
                 load_model_name=None,
                 load_weights_name=None,
                 trainable=True):

        self.input_shape = input_shape
        self.load_model_name = load_model_name
        self.load_weights_name = load_weights_name
        self.trainable = trainable
        self.model_name = model_name
        self.learning_rate = learning_rate

        self.model_save_name = None

        self.model_save_path = os.path.join(MODEL_SAVE_PATH, model_name)
        os.makedirs(self.model_save_path, exist_ok=True)

        self.model = None

        if load_model_name is not None:
            path = os.path.join(self.model_save_path, load_model_name)
            if os.path.exists(path):
                self.model = load_model(path)

        if self.model is None:
            self.model = self.create_model()

        if load_weights_name is not None:
            path = os.path.join(self.model_save_path, load_weights_name)
            if os.path.exists(path):
                self.model.load_weights(path)

        self.model.trainable = trainable
        self.model.summary()

        if trainable:
            loss, metrics = self.create_loss()
            self.model.compile(loss=loss, optimizer=Adam(learning_rate=self.learning_rate), metrics=metrics)

    def save(self, name):
        """
        保存模型
        :return:
        """
        name = f'{self.model_name}-{name}.h5'
        path = os.path.join(self.model_save_path, name)
        self.model.save(path)
        self.model_save_name = name
        return name

    def __save_history__(self, name, loss_list):
        path = os.path.join(self.model_save_path, f'{self.model_name}-{name}.png')
        fig = plt.figure()
        epochs = range(1, len(loss_list) + 1)
        plt.plot(epochs, loss_list, label='Loss')
        plt.title(f'{self.model_name} Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(path)
        plt.close(0)

    def fit_img_path(self, X, Y, epochs=10, batch_size=64, to_float=False):
        """
        模型训练，自定义图像加载时机，减少资源使用率，以防训练过程内存不足
        :param X: 图像路径集合
        :param Y: 标签集合
        :param epochs: 训练次数
        :param batch_size: 批数量
        :param to_float:
        :return: model_name
        """
        batch_count = int(len(X) / batch_size)
        if len(X) % batch_size != 0:
            batch_count += 1

        mse_list = []
        loss_list = []
        name = ''
        for epoch in range(epochs):
            mse_sum = 0
            loss_sum = 0
            start_time = time.perf_counter()
            for batch in range(batch_count):
                start_size = batch * batch_size
                end_size = start_size + batch_size
                x_train = X[start_size: end_size]
                y_train = Y[start_size: end_size]
                # 边训练边读取图像
                x_train = np.array([dataLoad.read_image(path, to_float) for path in x_train])

                history = self.model.fit(x_train, y_train, workers=10, verbose=0)
                mse_sum += history.history['mse'][0]
                loss_sum += history.history['loss'][0]

            mse = mse_sum / batch_count
            loss = loss_sum / batch_count
            mse_list.append(mse)
            loss_list.append(loss)
            end_time = time.perf_counter()
            print(f'{epoch + 1}/{epochs}\tloss:{loss:.4f}\tmse:{mse:.4f}\ttime:{end_time - start_time:.4f}')

            name = f'{time.strftime("%m%d%H%M")}-e({epochs})-b({batch_size})-eta({self.learning_rate})-loss({loss:.4f})'
            if (epoch + 1) % 10 == 0 and epoch < epochs - 1:
                # 每10轮保存模型
                self.__save_history__(name, loss_list)
                self.save(name)

        self.__save_history__(name, loss_list)
        return self.save(name)

    def fit_img_array(self, X, Y, epochs=10, batch_size=64):
        """
        模型训练，预先读取图像矩阵，消耗资源提升速度
        :param X: 图像矩阵集合
        :param Y: 标签集合
        :param epochs: 训练次数
        :param batch_size: 批数量
        :return: model_name
        """
        history = self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, workers=10, verbose=1)
        loss_list = history.history['loss']
        name = f'{time.strftime("%m%d%H%M")}-e({epochs})-b({batch_size})-eta({self.learning_rate})-loss({loss_list[-1]:.4f})'
        self.__save_history__(name, loss_list)
        return self.save(name)

    def predict_img_path(self, X, to_float=False) -> pd.DataFrame:
        """
        预测，根据图像路径预测
        :param X: 图像路径集合
        :param to_float:
        :return:
        """
        predict_y = np.array([self.model.predict(np.array([dataLoad.read_image(x, to_float)])).squeeze() for x in X])
        return pd.DataFrame(predict_y, columns=['predict'])

    def predict_img_array(self, X) -> pd.DataFrame:
        """

        :param X:
        :return:
        """
        y = np.array(self.model.predict(X)).squeeze()
        return pd.DataFrame(pd.Series(y, name='predict'))

    @abc.abstractmethod
    def create_model(self):
        """
        创建深度神经网络模型
        :return:
        """
        pass

    @abc.abstractmethod
    def create_loss(self):
        """
        定义模型使用的损失函数
        :return: loss, metrics
        """
        loss = 'mean_squared_error'
        metrics = ['mse']
        return loss, metrics
