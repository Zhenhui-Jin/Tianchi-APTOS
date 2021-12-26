from model import training, predict
from processing import data_factory


def train_image():
    eta = 0.00001
    epochs = 200
    model_load_path = 'model/ckpt/ModelAll-all/202112250931/ModelAll-all.pt'

    model_load_path = training.train_image(
        epochs=epochs,
        learning_rate=eta,
        model_load_path=model_load_path)

    print(model_load_path)


def train_csv():
    eta = 0.0001
    epochs = 500
    model_load_path = 'model/ckpt/Model-CSV/202112260744/Model-CSV.pt'

    model_load_path = training.train_csv(
        epochs=epochs,
        learning_rate=eta,
        model_load_path=model_load_path)

    print(model_load_path)


if __name__ == '__main__':
    # data_factory.processing_data()

    # train_image()

    train_csv()

    # model_image_path = 'model/ckpt/Model-Image/202112252342/Model-Image.pt'
    # model_csv_path = 'model/ckpt/Model-CSV/202112252318/Model-CSV.pt'
    # predict.predict(model_image_path, model_csv_path)
