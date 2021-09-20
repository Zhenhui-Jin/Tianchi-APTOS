import os

root_path = os.path.dirname(__file__)

# 数据集保存路径
DATA_PATH = os.path.join(root_path, 'data')

# 训练数据集
TRAIN_DATA_FILE = os.path.join(DATA_PATH, 'Train_set')

# 预测数据集
TEST_DATA_FILE = os.path.join(DATA_PATH, 'Test_set')

# keras 模型保存路径
KERAS_MODEL_SAVE_PATH = os.path.join(root_path, 'model', 'file', 'keras')
os.makedirs(KERAS_MODEL_SAVE_PATH, exist_ok=True)

# keras 模型预测结果保存路径
KERAS_PREDICT_SAVE_PATH = os.path.join(root_path, 'project', 'result', 'keras')
os.makedirs(KERAS_PREDICT_SAVE_PATH, exist_ok=True)
