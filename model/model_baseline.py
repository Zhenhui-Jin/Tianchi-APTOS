import os

import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization

from config import KERAS_MODEL_RESULT_SAVE_PATH
from data_load import dataLoad
from model_base import BaseModel


class CSTModel(BaseModel):

    def __init__(self,
                 input_shape=(500, 764, 3),
                 learning_rate=1e-3,
                 load_model_name=None,
                 load_weights_name=None,
                 trainable=True):
        super(CSTModel, self).__init__('CSTModel',
                                       input_shape=input_shape,
                                       learning_rate=learning_rate,
                                       load_model_name=load_model_name, load_weights_name=load_weights_name,
                                       trainable=trainable)
        self.padding = 'same'
        self.pool_size = (2, 2)
        self.strides = (2, 2)
        self.kernel_size = (3, 3)

    def create_model(self):
        """
        创建模型
        """
        model = Sequential()
        model.add(Conv2D(filters=16,
                         kernel_size=self.kernel_size,
                         input_shape=self.input_shape,
                         activation='relu',
                         strides=self.strides,
                         padding=self.padding))
        model.add(MaxPooling2D(pool_size=self.pool_size, padding=self.padding))
        model.add(BatchNormalization())

        model.add(Conv2D(filters=32,
                         kernel_size=self.kernel_size,
                         activation='relu',
                         strides=self.strides,
                         padding=self.padding))
        model.add(MaxPooling2D(pool_size=self.pool_size, padding=self.padding))
        model.add(BatchNormalization())

        model.add(Conv2D(filters=64,
                         kernel_size=self.kernel_size,
                         activation='relu',
                         strides=self.strides,
                         padding=self.padding))
        model.add(MaxPooling2D(pool_size=self.pool_size, padding=self.padding))
        model.add(BatchNormalization())

        model.add(Conv2D(filters=128,
                         kernel_size=self.kernel_size,
                         activation='relu',
                         strides=self.strides,
                         padding=self.padding))
        model.add(MaxPooling2D(pool_size=self.pool_size, padding=self.padding))
        model.add(BatchNormalization())

        model.add(Conv2D(filters=256,
                         kernel_size=self.kernel_size,
                         activation='relu',
                         strides=self.strides,
                         padding=self.padding))
        model.add(MaxPooling2D(pool_size=self.pool_size, padding=self.padding))
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


def cst_model_fit(model_name, epochs=10):
    train_data = dataLoad.get_train_data_cst_all(read_img=True)
    X = np.array(train_data['feature'].tolist())
    print('train:', X.shape)
    Y = train_data['label']
    # for i in range(10):
    model = CSTModel(learning_rate=1e-9, load_model_name=model_name)
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
    model_name = 'CSTModel-09212209-e(100)-b(64)-eta(1e-06)-loss(480.9583).h5'
    cst_model_fit(model_name, epochs=100)
