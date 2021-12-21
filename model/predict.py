import pandas as pd

import config
from model.model import ImageModel
from model_config import ImageConfig


def predict_image(model_name, model_load_path, **kwargs):
    print('predict_image')
    test_data = pd.read_csv(config.PROCESSED_TEST_CSV_PATH)
    image_config = ImageConfig(name=model_name, model_load_path=model_load_path, training=False, **kwargs)
    image_model = ImageModel(image_config)
    predict_data = image_model.eval(test_data)
    predict_data.to_csv(image_config.model_result_path, index=False)
