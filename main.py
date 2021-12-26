from model import training, predict
from processing import data_factory


def train_image():
    eta = 0.001
    epochs = 100

    model_path = {
        'Model-Image-After': (1, ''),
        'Model-Image-Before': (0, ''),
        'Model-Image-All': (None, ''),
    }
    for eta, epochs in zip([0.001, 0.001, 0.001], [100, 100, 100]):
        for key, (after, path), in model_path.items():
            path = training.train_image(
                after=after,
                epochs=epochs,
                learning_rate=eta,
                model_load_path=path)
            model_path[key] = (after, path)

        print(model_path)


def train_csv():
    # model_load_path = 'model/ckpt/Model-CSV/202112262210/Model-CSV.pt'
    model_load_path = ''
    # for eta, epochs in zip([0.0001, 0.0001], [2000, 5000]):
    for eta, epochs in zip([0.0001, 0.0001], [4000, 5000]):
        model_load_path = training.train_csv(
            epochs=epochs,
            learning_rate=eta,
            model_load_path=model_load_path)

        print(model_load_path)


def start_process():
    select = ['请输入数字:\n',
              ' (0).processing_data \n',
              ' (1).train_image\n',
              ' (2).train_csv\n',
              ' (3).predict\n',
              ' (4).train_csv+train_image\n']
    item = input("".join(select))
    if item == '0':
        data_factory.processing_data()
    elif item == '1':
        train_image()
    elif item == '2':
        train_csv()
    elif item == '3':
        model_image_path = 'model/ckpt/Model-Image/202112260010/Model-Image.pt'
        model_csv_path = 'model/ckpt/Model-CSV/202112262235/Model-CSV.pt'
        predict.predict(model_image_path, model_csv_path)
    elif item == '4':
        train_csv()
        train_image()
    else:
        start_process()


if __name__ == '__main__':
    start_process()
