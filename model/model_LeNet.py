from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AvgPool2D

from model_base import BaseModel

"""
loss 一直降不下来
"""


class LeNetModel(BaseModel):

    def __init__(self,
                 input_shape=(500, 764, 3),
                 learning_rate=0.001,
                 load_model_name=None,
                 load_weights_name=None,
                 trainable=True):
        super(LeNetModel, self).__init__('LeNet', input_shape=input_shape, learning_rate=learning_rate,
                                         load_model_name=load_model_name, load_weights_name=load_weights_name,
                                         trainable=trainable)

    def create_model(self):
        """
        创建模型
        :return:
        """
        model = Sequential()
        model.add(Conv2D(filters=2, kernel_size=2, input_shape=self.input_shape, activation='sigmoid', padding='same'))
        model.add(Conv2D(filters=6, kernel_size=5, activation='sigmoid', padding='same'))
        model.add(AvgPool2D(pool_size=2, strides=2))
        model.add(Conv2D(filters=16, kernel_size=5, activation='sigmoid'))
        model.add(AvgPool2D(pool_size=2, strides=2))
        model.add(Flatten())
        model.add(Dense(120, activation='sigmoid'))
        model.add(Dense(84, activation='sigmoid'))
        model.add(Dense(1))

        return model
