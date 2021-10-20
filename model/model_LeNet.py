from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Flatten, Conv2D, AvgPool2D

from model_base import BaseModel
from data_load import DataLoad


class LeNetModel(BaseModel):

    def __init__(self,
                 model_name='LeNet',
                 input_shape=(500, 764, 3),
                 learning_rate=0.001,
                 load_model_name=None,
                 load_weights_name=None,
                 trainable=True,
                 dataLoad: DataLoad = None,
                 ):
        super(LeNetModel, self).__init__(model_name=model_name, input_shape=input_shape, learning_rate=learning_rate,
                                         load_model_name=load_model_name, load_weights_name=load_weights_name,
                                         trainable=trainable, dataLoad=dataLoad)

    def create_model(self):
        """
        创建模型
        :return:
        """
        model = Sequential()
        model.add(InputLayer(input_shape=self.input_shape))
        model.add(Conv2D(filters=6, kernel_size=5, activation='relu', padding='same'))
        model.add(AvgPool2D(pool_size=2, strides=2))
        model.add(Conv2D(filters=16, kernel_size=5, activation='relu'))
        model.add(AvgPool2D(pool_size=2, strides=2))
        model.add(Flatten())
        model.add(Dense(120, activation='relu'))
        model.add(Dense(84, activation='relu'))
        model.add(Dense(1))

        return model
