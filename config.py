import os

root_path = os.path.dirname(__file__)

# 数据集保存路径
DATA_PATH = os.path.join(root_path, 'data')

# 训练数据集
TRAIN_DATA_FILE = os.path.join(DATA_PATH, 'Train_set')
TRAIN_DATA_FILE_NEW = os.path.join(DATA_PATH, 'train')
# 预测数据集
TEST_DATA_FILE = os.path.join(DATA_PATH, 'Test_set')
TEST_DATA_FILE_NEW = os.path.join(DATA_PATH, 'test')
os.makedirs(TRAIN_DATA_FILE_NEW, exist_ok=True)
os.makedirs(TEST_DATA_FILE_NEW, exist_ok=True)

# CST训练数据集
CST_TRAIN_DATA_FILE = os.path.join(DATA_PATH, 'CST', 'train')
# CST预测数据集
CST_TEST_DATA_FILE = os.path.join(DATA_PATH, 'CST', 'test')
# os.makedirs(CST_TRAIN_DATA_FILE, exist_ok=True)
# os.makedirs(CST_TEST_DATA_FILE, exist_ok=True)

# 模型文件保存路径
MODEL_SAVE_PATH = os.path.join(root_path, 'model', 'file')

# keras 模型保存路径
KERAS_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, 'keras')
# 模型训练结果保存路径
MODEL_RESULT_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, 'result')
os.makedirs(MODEL_RESULT_SAVE_PATH, exist_ok=True)

# 模型预测结果保存路径
PREDICT_SAVE_PATH = os.path.join(root_path, 'predict', 'result')
os.makedirs(PREDICT_SAVE_PATH, exist_ok=True)
