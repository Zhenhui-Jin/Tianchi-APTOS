from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import add, Input, InputLayer, Dense, Flatten, Conv2D, AvgPool2D, ZeroPadding2D

from model_base import BaseModel

"""
loss 一直降不下来
"""


class ResidualNet(BaseModel):

    def __init__(self,
                 input_shape=(500, 764, 3),
                 learning_rate=0.001,
                 load_model_name=None,
                 load_weights_name=None,
                 trainable=True):
        super(ResidualNet, self).__init__('ResidualNet', input_shape=input_shape, learning_rate=learning_rate,
                                          load_model_name=load_model_name, load_weights_name=load_weights_name,
                                          trainable=trainable)

    def create_model(self):
        """
        创建模型
        :return:
        """
        X_input = Input(shape=self.input_shape)
        X = X_input
        X = Conv2D(filters=96, kernel_size=11, padding='same', activation='relu')(X)
        X = Conv2D(filters=3, kernel_size=11, padding='same', activation='relu')(X)
        X = add([X_input, X])
        X = Flatten()(X)
        fc = Dense(units=1000)(X)
        model = Model(input=X_input, output=fc, name=self.model_name)

        return model


if __name__ == '__main__':
    ResidualNet(input_shape=(224, 224, 3)).save('')
