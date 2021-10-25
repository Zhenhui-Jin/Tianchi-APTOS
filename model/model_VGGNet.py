from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, InputLayer, Dense, Flatten, Conv2D, MaxPooling2D

from tensorflow.keras.models import Model

from data_load import DataLoad
from model_base import BaseModel


class VGGNet11(BaseModel):

    def __init__(self,
                 model_name='VGGNet11',
                 input_shape=(500, 764, 3),
                 learning_rate=1e-3,
                 load_model_name=None,
                 load_weights_name=None,
                 trainable=True,
                 dataLoad: DataLoad = None):
        super(VGGNet11, self).__init__(model_name=model_name,
                                       input_shape=input_shape,
                                       learning_rate=learning_rate,
                                       load_model_name=load_model_name, load_weights_name=load_weights_name,
                                       trainable=trainable, dataLoad=dataLoad)

    def create_model(self):
        X_input = Input(shape=self.input_shape)
        X = X_input

        X = Conv2D(filters=16, kernel_size=3, activation='relu', padding='same')(X)
        X = MaxPooling2D(pool_size=2, padding='same')(X)

        X = Conv2D(filters=32, kernel_size=3, activation='relu', padding='same')(X)
        X = MaxPooling2D(pool_size=2, padding='same')(X)

        X = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(X)
        X = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(X)
        X = MaxPooling2D(pool_size=2, padding='same')(X)

        X = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(X)
        X = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(X)
        X = MaxPooling2D(pool_size=2, padding='same')(X)

        X = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(X)
        X = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(X)
        X = MaxPooling2D(pool_size=2, padding='same')(X)

        X = Flatten()(X)
        X = Dense(units=64, activation='relu')(X)
        X = Dense(units=32, activation='relu')(X)
        fc = Dense(units=1)(X)

        model = Model(inputs=X_input, outputs=fc, name=self.model_name)

        return model


class VGGNet16(BaseModel):

    def __init__(self,
                 input_shape=(500, 764, 3),
                 learning_rate=1e-3,
                 load_model_name=None,
                 load_weights_name=None,
                 trainable=True):
        super(VGGNet16, self).__init__('VGGNet16',
                                       input_shape=input_shape,
                                       learning_rate=learning_rate,
                                       load_model_name=load_model_name, load_weights_name=load_weights_name,
                                       trainable=trainable)

    def create_model(self):
        X_input = Input(shape=self.input_shape)
        X = X_input

        X = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(X)
        X = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(X)
        X = MaxPooling2D(pool_size=2, padding='same')(X)

        X = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(X)
        X = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(X)
        X = MaxPooling2D(pool_size=2, padding='same')

        X = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(X)
        X = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(X)
        X = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(X)
        X = MaxPooling2D(pool_size=2, padding='same')

        X = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same')(X)
        X = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same')(X)
        X = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same')(X)
        X = MaxPooling2D(pool_size=2, padding='same')

        X = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same')(X)
        X = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same')(X)
        X = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same')(X)
        X = MaxPooling2D(pool_size=2, padding='same')

        X = Flatten()(X)
        X = Dense(units=4096, activation='relu')(X)
        X = Dense(units=4096, activation='relu')(X)
        fc = Dense(units=1)(X)

        model = Model(input=X_input, output=fc, name=self.model_name)

        return model


class VGGNet19(BaseModel):

    def __init__(self,
                 input_shape=(500, 764, 3),
                 learning_rate=1e-3,
                 load_model_name=None,
                 load_weights_name=None,
                 trainable=True):
        super(VGGNet19, self).__init__('VGGNet19',
                                       input_shape=input_shape,
                                       learning_rate=learning_rate,
                                       load_model_name=load_model_name, load_weights_name=load_weights_name,
                                       trainable=trainable)

    def create_model(self):
        X_input = Input(shape=self.input_shape)
        X = X_input

        X = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(X)
        X = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(X)
        X = MaxPooling2D(pool_size=2, padding='same')(X)

        X = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(X)
        X = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(X)
        X = MaxPooling2D(pool_size=2, padding='same')(X)

        X = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(X)
        X = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(X)
        X = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(X)
        X = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(X)
        X = MaxPooling2D(pool_size=2, padding='same')(X)

        X = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same')(X)
        X = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same')(X)
        X = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same')(X)
        X = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same')(X)
        X = MaxPooling2D(pool_size=2, padding='same')(X)

        X = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same')(X)
        X = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same')(X)
        X = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same')(X)
        X = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same')(X)
        X = MaxPooling2D(pool_size=2, padding='same')(X)

        X = Flatten()(X)
        X = Dense(units=4096, activation='relu')(X)
        X = Dense(units=4096, activation='relu')(X)
        fc = Dense(units=1)(X)

        model = Model(input=X_input, output=fc, name=self.model_name)

        return model
