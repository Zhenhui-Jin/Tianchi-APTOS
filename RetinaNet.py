from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.layers import Conv2D

if __name__ == '__main__':
    net = ResNet101(include_top=False, input_shape=[None, None, 3], weights=None)

    net.save('ResNet101.h5')
