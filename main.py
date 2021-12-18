from model import training
from processing import data_factory

if __name__ == '__main__':
    # data_factory.processing_data()
    training.train_image()
    training.train_image(after=0)
    training.train_image(after=0, final=0)
    training.train_image(after=0, final=1)
    training.train_image(after=1)
    training.train_image(after=1, final=0)
    training.train_image(after=1, final=1)
