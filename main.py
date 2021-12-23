from model import training, predict
from processing import data_factory


def train():
    model_path = {
        'ModelAll-all': 'model/ckpt/ModelAll-all/202112230816/ModelAll-all.pt',
        'ModelAll-final': 'model/ckpt/ModelAll-final/202112231139/ModelAll-final.pt',
        'ModelAll-preliminary': 'model/ckpt/ModelAll-preliminary/202112231201/ModelAll-preliminary.pt',
        'ModelBefore-all': 'model/ckpt/ModelBefore-all/202112231512/ModelBefore-all.pt',
        'ModelBefore-preliminary': 'model/ckpt/ModelBefore-preliminary/202112231659/ModelBefore-preliminary.pt',
        'ModelBefore-final': 'model/ckpt/ModelBefore-final/202112231840/ModelBefore-final.pt',
        'ModelAfter-all': 'model/ckpt/ModelAfter-all/202112231855/ModelAfter-all.pt',
        'ModelAfter-preliminary': 'model/ckpt/ModelAfter-preliminary/202112232044/ModelAfter-preliminary.pt',
        'ModelAfter-final': 'model/ckpt/ModelAfter-final/202112232235/ModelAfter-final.pt'}

    eta = 0.001

    model_path['ModelAll-all'] = training.train_image(
        epochs=200,
        learning_rate=eta,
        model_load_path=model_path.get('ModelAll-all'))

    print(model_path)

    model_path['ModelAll-all'] = training.train_image(
        epochs=100,
        learning_rate=eta,
        model_load_path=model_path.get('ModelAll-all'))

    # model_path['ModelAll-final'] = training.train_image(
    #     final=1,
    #     learning_rate=eta,
    #     model_load_path=model_path.get('ModelAll-final'))

    # model_path['ModelAll-preliminary'] = training.train_image(
    #     final=0,
    #     learning_rate=eta,
    #     model_load_path=model_path.get('ModelAll-preliminary'))

    # model_path['ModelBefore-all'] = training.train_image(
    #     after=0,
    #     learning_rate=eta,
    #     model_load_path=model_path.get('ModelBefore-all'))

    # model_path['ModelBefore-preliminary'] = training.train_image(
    #     after=0, final=0,
    #     learning_rate=eta,
    #     model_load_path=model_path.get('ModelBefore-preliminary'))

    # model_path['ModelBefore-final'] = training.train_image(
    #     after=0, final=1,
    #     learning_rate=eta,
    #     model_load_path=model_path.get('ModelBefore-final'))

    # model_path['ModelAfter-all'] = training.train_image(
    #     after=1,
    #     learning_rate=eta,
    #     model_load_path=model_path.get('ModelAfter-all'))

    # model_path['ModelAfter-preliminary'] = training.train_image(
    #     after=1, final=0,
    #     learning_rate=eta,
    #     model_load_path=model_path.get('ModelAfter-preliminary'))

    # model_path['ModelAfter-final'] = training.train_image(
    #     after=1, final=1,
    #     learning_rate=eta,
    #     model_load_path=model_path.get('ModelAfter-final'))

    print(model_path)


if __name__ == '__main__':
    # data_factory.processing_data()

    train()

    # predict.predict_image('ModelAll-all', 'model/ckpt/ModelAll-all/202112181853/ModelAll-all.pt')
