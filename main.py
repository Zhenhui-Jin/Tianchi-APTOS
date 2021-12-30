from model import training, predict
from processing import data_factory


def train_image():
    model_path = {
        # 'Model-Image-All': (None, 'model/ckpt/Model-Image-All/202112301955/Model-Image-All.pt'),
        'Model-Image-After': (1, 'model/ckpt/Model-Image-After/202112301618/Model-Image-After.pt'),
        'Model-Image-Before': (0, 'model/ckpt/Model-Image-Before/202112301803/Model-Image-Before.pt')}

    for eta, epochs in zip([0.0001, 0.0001, 0.0001], [50, 50, 50]):
        for key, (after, path), in model_path.items():
            path = training.train_image(
                    after=after,
                    epochs=epochs,
                    learning_rate=eta,
                    model_load_path=path)
            model_path[key] = (after, path)

        print(model_path)


def train_csv():
    model_load_path = 'model/ckpt/Model-CSV/202112310622/Model-CSV.pt'

    # model_load_path = ''
    for eta, epochs in zip([1e-5, 1e-5, 1e-5], [2000]):
        model_load_path = training.train_csv(
            epochs=epochs,
            learning_rate=eta,
            model_load_path=model_load_path)

        print(model_load_path)


def train_csv_image():
    train_csv()
    train_image()


def start_process():
    fn = [('processing_data', data_factory.processing_data),
          ('train_image', train_image),
          ('train_csv', train_csv),
          ('train_csv + train_image', train_csv_image),
          ('predict_before_after', predict.predict_before_after),
          ('predict_all', predict.predict_all),
          ]

    select = ['请输入数字']

    for index, (key, f) in zip(range(len(fn)), fn):
        select.append(f'{index} -> {key}')

    item = input('\n'.join(select) + '\n')

    model_csv_path = 'model/ckpt/Model-CSV/202112302328/Model-CSV.pt'
    model_image_path_before = 'model/ckpt/Model-Image-Before/202112301803/Model-Image-Before.pt'
    model_image_path_after = 'model/ckpt/Model-Image-After/202112301618/Model-Image-After.pt'
    model_image_path_all = 'model/ckpt/Model-Image-All/202112301955/Model-Image-All.pt'

    try:
        index = int(item)
        key = fn[index][0]
        F = fn[index][1]
        if key == 'predict_all':
            F(model_image_path_all, model_csv_path)
        elif key == 'predict_before_after':
            F(model_image_path_before, model_image_path_after, model_csv_path)
        else:
            F()
    except:
        start_process()


if __name__ == '__main__':
    start_process()
