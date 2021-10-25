from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Flatten, Conv2D, MaxPooling2D

from model_base import BaseModel


class AlexNet(BaseModel):

    def __init__(self,
                 input_shape=(500, 764, 3),
                 learning_rate=1e-3,
                 load_model_name=None,
                 load_weights_name=None,
                 trainable=True):
        super(AlexNet, self).__init__('AlexNetModel',
                                      input_shape=input_shape,
                                      learning_rate=learning_rate,
                                      load_model_name=load_model_name, load_weights_name=load_weights_name,
                                      trainable=trainable)

    def create_model(self):
        model = Sequential()

        model.add(InputLayer(input_shape=self.input_shape))

        model.add(Conv2D(filters=96, kernel_size=11, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=2, padding='same'))

        model.add(Conv2D(filters=256, kernel_size=5, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=2, padding='same'))

        model.add(Conv2D(filters=384, kernel_size=3, activation='relu', padding='same'))
        model.add(Conv2D(filters=384, kernel_size=3, activation='relu', padding='same'))
        model.add(Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=2, padding='same'))

        model.add(Flatten())
        model.add(Dense(units=4096, activation='relu'))
        model.add(Dense(units=4096, activation='relu'))
        model.add(Dense(units=1))

        return model
