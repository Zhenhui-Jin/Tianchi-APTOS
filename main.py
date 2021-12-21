from model import training, predict
from processing import data_factory

if __name__ == '__main__':
    # data_factory.processing_data()

    model_path = {
        'ModelAll-all': 'model/ckpt/ModelAll-all/202112201402/ModelAll-all.pt',
        'ModelAll-final': 'model/ckpt/ModelAll-final/202112202321/ModelAll-final.pt',
        'ModelAll-preliminary': 'model/ckpt/ModelAll-preliminary/202112202126/ModelAll-preliminary.pt',
        'ModelBefore-all': 'model/ckpt/ModelBefore-all/202112201602/ModelBefore-all.pt',
        'ModelBefore-preliminary': 'model/ckpt/ModelBefore-preliminary/202112201705/ModelBefore-preliminary.pt',
        'ModelBefore-final': 'model/ckpt/ModelBefore-final/202112201803/ModelBefore-final.pt',
        'ModelAfter-all': 'model/ckpt/ModelAfter-all/202112201811/ModelAfter-all.pt',
        'ModelAfter-preliminary': 'model/ckpt/ModelAfter-preliminary/202112201915/ModelAfter-preliminary.pt',
        'ModelAfter-final': 'model/ckpt/ModelAfter-final/202112202015/ModelAfter-final.pt'
    }

    model_path = {
        'ModelAll-all': 'D:\\Development\\Workspace\\AI\\Tianchi-APTOS\\model\\ckpt\\ModelAll-all\\202112210759\\ModelAll-all.pt',
        'ModelAll-final': 'D:\\Development\\Workspace\\AI\\Tianchi-APTOS\\model\\ckpt\\ModelAll-final\\202112211201\\ModelAll-final.pt',
        'ModelAll-preliminary': 'D:\\Development\\Workspace\\AI\\Tianchi-APTOS\\model\\ckpt\\ModelAll-preliminary\\202112211225\\ModelAll-preliminary.pt',
        'ModelBefore-all': 'D:\\Development\\Workspace\\AI\\Tianchi-APTOS\\model\\ckpt\\ModelBefore-all\\202112211612\\ModelBefore-all.pt',
        'ModelBefore-preliminary': 'D:\\Development\\Workspace\\AI\\Tianchi-APTOS\\model\\ckpt\\ModelBefore-preliminary\\202112211817\\ModelBefore-preliminary.pt',
        'ModelBefore-final': 'D:\\Development\\Workspace\\AI\\Tianchi-APTOS\\model\\ckpt\\ModelBefore-final\\202112212014\\ModelBefore-final.pt',
        'ModelAfter-all': 'D:\\Development\\Workspace\\AI\\Tianchi-APTOS\\model\\ckpt\\ModelAfter-all\\202112212031\\ModelAfter-all.pt',
        'ModelAfter-preliminary': 'D:\\Development\\Workspace\\AI\\Tianchi-APTOS\\model\\ckpt\\ModelAfter-preliminary\\202112212242\\ModelAfter-preliminary.pt',
        'ModelAfter-final': 'D:\\Development\\Workspace\\AI\\Tianchi-APTOS\\model\\ckpt\\ModelAfter-final\\202112220042\\ModelAfter-final.pt'}
    eta = 0.001

    model_path['ModelAll-all'] = training.train_image(
        learning_rate=eta,
        model_load_path=model_path.get('ModelAll-all'))

    model_path['ModelAll-final'] = training.train_image(
        final=1,
        learning_rate=eta,
        model_load_path=model_path.get('ModelAll-final'))

    model_path['ModelAll-preliminary'] = training.train_image(
        final=0,
        learning_rate=eta,
        model_load_path=model_path.get('ModelAll-preliminary'))

    model_path['ModelBefore-all'] = training.train_image(
        after=0,
        learning_rate=eta,
        model_load_path=model_path.get('ModelBefore-all'))

    model_path['ModelBefore-preliminary'] = training.train_image(
        after=0, final=0,
        learning_rate=eta,
        model_load_path=model_path.get('ModelBefore-preliminary'))

    model_path['ModelBefore-final'] = training.train_image(
        after=0, final=1,
        learning_rate=eta,
        model_load_path=model_path.get('ModelBefore-final'))

    model_path['ModelAfter-all'] = training.train_image(
        after=1,
        learning_rate=eta,
        model_load_path=model_path.get('ModelAfter-all'))

    model_path['ModelAfter-preliminary'] = training.train_image(
        after=1, final=0,
        learning_rate=eta,
        model_load_path=model_path.get('ModelAfter-preliminary'))

    model_path['ModelAfter-final'] = training.train_image(
        after=1, final=1,
        learning_rate=eta,
        model_load_path=model_path.get('ModelAfter-final'))

    print(model_path)

    # predict.predict_image('ModelAll-all', 'model/ckpt/ModelAll-all/202112181853/ModelAll-all.pt')
