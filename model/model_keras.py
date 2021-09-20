from data_load import dataLoad
from config import KERAS_MODEL_SAVE_PATH

import os
import time
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import load_model


class Model:

    def __init__(self,
                 input_shape=(500, 764, 3),
                 kernel_size=(3, 3),
                 strides=(2, 2),
                 pool_size=(2, 2),
                 padding='same',
                 load_model_name=None,
                 load_weights_name=None,
                 trainable=True):

        self.model = None

        if load_model_name:
            path = os.path.join(KERAS_MODEL_SAVE_PATH, load_model_name)
            if os.path.exists(path):
                self.model = load_model(path)

        if self.model is None:
            self.model = self.create_model(input_shape, kernel_size, padding, pool_size, strides)

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
        path = os.path.join(KERAS_MODEL_SAVE_PATH, f'model-{name}-weights.h5')
        self.model.save_weights(path)
        model_name = f'model-{name}.h5'
        path = os.path.join(KERAS_MODEL_SAVE_PATH, model_name)
        self.model.save(path)
        return model_name

    def __save_history__(self, name, history):
        print(history)
        history_path = os.path.join(KERAS_MODEL_SAVE_PATH, 'history')
        os.makedirs(history_path, exist_ok=True)

        path = os.path.join(history_path, f'model-{name}-MSE.png')
        fig = plt.figure()
        # 绘制训练 & 验证的准确率值
        plt.plot(history.history['mse'], label='Train')
        plt.plot(history.history['val_mse'], label='Test')
        plt.title('Model MSE')
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(path)
        plt.close(0)

        fig = plt.figure()
        path = os.path.join(history_path, f'model-{name}-Loss.png')
        # 绘制训练 & 验证的损失值
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Test')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(path)
        plt.close(0)

    def fit(self, x_train, y_train, epochs=10, batch_size=64, learning_rate=1e-3):
        """
        模型训练
        :param x_train:
        :param y_train:
        :param epochs:
        :param batch_size:
        :param learning_rate:
        :return:
        """
        adam = Adam(learning_rate=learning_rate)
        self.model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])

        history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, workers=10, verbose=1)

        name = f'{time.strftime("%m%d%H%M")}-e{epochs}-b{batch_size}-eta{learning_rate}'
        self.__save_history__(name, history)
        return self.save(name)


if __name__ == '__main__':
    train_data, label = dataLoad.get_pre_cst_train_data()
    print(train_data.shape)
    model_name = 'model-09201112-e100-b64-eta0.0001.h5'
    for epo in range(5):
        model_name = Model(input_shape=(500, 764, 3), load_model_name=model_name).fit(train_data, label, epochs=100, learning_rate=1e-4)
