from data_load import dataLoad
from config import KERAS_MODEL_SAVE_PATH, KERAS_MODEL_RESULT_SAVE_PATH, KERAS_MODEL_HISTORY_SAVE_PATH, \
    KERAS_MODEL_LEARNING_RATE_SAVE_PATH

import json
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.optimizers.schedules import serialize, deserialize, PolynomialDecay
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import load_model


class CSTModel:

    def __init__(self,
                 input_shape=(500, 764, 3),
                 kernel_size=(3, 3),
                 strides=(2, 2),
                 pool_size=(2, 2),
                 learning_rate=1e-3,
                 padding='same',
                 load_model_name=None,
                 load_weights_name=None,
                 trainable=True):

        self.learning_rate = learning_rate
        self.model = None

        if load_model_name:
            path = os.path.join(KERAS_MODEL_SAVE_PATH, load_model_name)
            if os.path.exists(path):
                self.model = load_model(path)

        if self.model is None:
            self.model = self.create_model(input_shape, kernel_size, padding, pool_size, strides)

        adam = Adam(learning_rate=self.learning_rate)
        self.model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])

        if load_weights_name:
            path = os.path.join(KERAS_MODEL_SAVE_PATH, load_weights_name)
            if os.path.exists(path):
                self.model.load_weights(path)

        self.model.trainable = trainable
        self.model.summary()

    @staticmethod
    def create_model(input_shape, kernel_size, padding, pool_size, strides):
        """
        创建模型
        :param input_shape:
        :param kernel_size:
        :param padding:
        :param pool_size:
        :param strides:
        :return:
        """
        model = Sequential()
        model.add(Conv2D(filters=16,
                         kernel_size=kernel_size,
                         input_shape=input_shape,
                         activation='relu',
                         strides=strides,
                         padding=padding))
        model.add(MaxPooling2D(pool_size=pool_size, padding=padding))
        model.add(BatchNormalization())

        model.add(Conv2D(filters=32,
                         kernel_size=kernel_size,
                         activation='relu',
                         strides=strides,
                         padding=padding))
        model.add(MaxPooling2D(pool_size=pool_size, padding=padding))
        model.add(BatchNormalization())

        model.add(Conv2D(filters=64,
                         kernel_size=kernel_size,
                         activation='relu',
                         strides=strides,
                         padding=padding))
        model.add(MaxPooling2D(pool_size=pool_size, padding=padding))
        model.add(BatchNormalization())

        model.add(Conv2D(filters=128,
                         kernel_size=kernel_size,
                         activation='relu',
                         strides=strides,
                         padding=padding))
        model.add(MaxPooling2D(pool_size=pool_size, padding=padding))
        model.add(BatchNormalization())

        model.add(Conv2D(filters=256,
                         kernel_size=kernel_size,
                         activation='relu',
                         strides=strides,
                         padding=padding))
        model.add(MaxPooling2D(pool_size=pool_size, padding=padding))
        model.add(BatchNormalization())

        model.add(Flatten())

        model.add(Dense(units=256))
        model.add(BatchNormalization())

        model.add(Dense(units=128))
        model.add(BatchNormalization())

        model.add(Dense(units=64))
        model.add(BatchNormalization())

        model.add(Dense(units=32))
        model.add(BatchNormalization())

        model.add(Dense(units=1))

        return model

    def save(self, name):
        """
        保存模型
        :return:
        """
        path = os.path.join(KERAS_MODEL_SAVE_PATH, f'CSTModel-{name}-weights.h5')
        self.model.save_weights(path)
        name = f'CSTModel-{name}.h5'
        path = os.path.join(KERAS_MODEL_SAVE_PATH, name)
        self.model.save(path)
        return name

    def __save_history__(self, name, loss_list):
        path = os.path.join(KERAS_MODEL_HISTORY_SAVE_PATH, f'CSTModel-{name}.png')
        fig = plt.figure()
        plt.plot(loss_list, label='Loss')
        plt.title('CSTModel Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(path)
        plt.close(0)

    def fit_img_path(self, X, Y, epochs=10, batch_size=64):
        """
        模型训练，自定义图像加载时机，减少资源使用率，以防训练过程内存不足
        :param X: 图像路径集合
        :param Y: 标签集合
        :param epochs: 训练次数
        :param batch_size: 批数量
        :return: model_name
        """
        batch_count = int(len(X) / batch_size)
        if len(X) % batch_size != 0:
            batch_count += 1

        mse_list = []
        loss_list = []
        loss = 0
        for epoch in range(epochs):
            print(f'{epoch + 1}/{epochs}')
            mse_sum = 0
            loss_sum = 0
            for batch in tqdm(range(batch_count)):
                start_size = batch * batch_size
                end_size = start_size + batch_size
                x_train = X[start_size: end_size]
                y_train = Y[start_size: end_size]
                # 边训练边读取图像
                x_train = np.array([dataLoad.read_image(path) for path in x_train])

                history = self.model.fit(x_train, y_train, workers=10, verbose=0)
                mse_sum += history.history['mse'][0]
                loss_sum += history.history['loss'][0]

            mse = mse_sum / batch_count
            loss = loss_sum / batch_count
            mse_list.append(mse)
            loss_list.append(loss)
            print(f'{epoch + 1}/{epochs}\tloss:{loss:.4f}\tmse:{mse:.4f}')

        name = f'{time.strftime("%m%d%H%M")}-e({epochs})-b({batch_size})-loss({loss:.4f})'
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
        name = f'{time.strftime("%m%d%H%M")}-e({epochs})-b({batch_size})-loss({loss_list[-1]:.4f})'
        self.__save_history__(name, loss_list)
        return self.save(name)

    def predict_img_path(self, X) -> pd.DataFrame:
        """
        预测，根据图像路径预测
        :param X: 图像路径集合
        :return:
        """

        def get_predict(x):
            return np.array(self.model.predict(np.array([dataLoad.read_image(x)]))).squeeze()

        return pd.DataFrame(pd.Series(X.apply(get_predict), name='predict'))

    def predict_img_array(self, X) -> pd.DataFrame:
        """

        :param X:
        :return:
        """
        y = np.array(self.model.predict(X)).squeeze()
        return pd.DataFrame(pd.Series(y, name='predict'))


def cst_model_fit(model_name, epochs=10):
    train_data = dataLoad.get_train_data_cst_all(read_img=True)
    X = np.array(train_data['feature'].tolist())
    print('train:', X.shape)
    Y = train_data['label']
    # for i in range(10):
    model = CSTModel(load_model_name=model_name)
    new_model_name = model.fit_img_array(X, Y, epochs=epochs)
    model_name = new_model_name

    predict_y = model.predict_img_array(X)
    predict_data = pd.concat([train_data.drop('feature', axis=1), predict_y], axis=1)
    predict_data.to_csv(os.path.join(KERAS_MODEL_RESULT_SAVE_PATH, f'train-{model_name}.csv'), index=False)

    test_data = dataLoad.get_test_data_cst_all(read_img=True)
    X_test = np.array(test_data['feature'].tolist())
    print('test:', X_test.shape)

    predict_y = model.predict_img_array(X_test)
    predict_data = pd.concat([test_data.drop('feature', axis=1), predict_y], axis=1)
    predict_data.to_csv(os.path.join(KERAS_MODEL_RESULT_SAVE_PATH, f'test-{model_name}.csv'), index=False)


if __name__ == '__main__':
    model_name = 'CSTModel-09212328-e(100)-b(64)-loss(750.0526).h5'
    cst_model_fit(model_name, epochs=1000)
