import os

root_path = os.path.dirname(__file__)

# 数据集根路径
DATA_PATH = os.path.join(root_path, 'data')

# 源数据路径
SOURCE_PATH = os.path.join(DATA_PATH, 'source')
# 初赛源数据
SOURCE_PATH_PRELIMINARY = os.path.join(SOURCE_PATH, 'preliminary')
# 初赛源测试图像数据
SOURCE_TEST_IMG_PATH_PRELIMINARY = os.path.join(SOURCE_PATH_PRELIMINARY, 'Test_set')
# 初赛源测试CSV数据
SOURCE_TEST_CSV_PATH_PRELIMINARY = os.path.join(SOURCE_TEST_IMG_PATH_PRELIMINARY, 'PreliminaryValidationSet_Info.csv')
# 初赛源训练图像数据
SOURCE_TRAIN_IMG_PATH_PRELIMINARY = os.path.join(SOURCE_PATH_PRELIMINARY, 'Train_set')
# 初赛源训练CSV数据
SOURCE_TRAIN_CSV_PATH_PRELIMINARY = os.path.join(SOURCE_TRAIN_IMG_PATH_PRELIMINARY, 'TrainingAnnotation.csv')

# 复赛源数据
SOURCE_PATH_FINAL = os.path.join(SOURCE_PATH, 'final')
# 复赛源测试图像数据
SOURCE_TEST_IMG_PATH_FINAL = os.path.join(SOURCE_PATH_FINAL, 'Test_set')
# 复赛源测试CSV数据
SOURCE_TEST_CSV_PATH_FINAL = os.path.join(SOURCE_PATH_FINAL, 'test_info.csv')
# 复赛源训练图像数据
SOURCE_TRAIN_IMG_PATH_FINAL = os.path.join(SOURCE_PATH_FINAL, 'Train_Set')
# 复赛源训练CASE CSV数据
SOURCE_TRAIN_CASE_CSV_PATH_FINAL = os.path.join(SOURCE_PATH_FINAL, 'train_anno_case.csv')
# 复赛源训练PIC CSV数据
SOURCE_TRAIN_PIC_CSV_PATH_FINAL = os.path.join(SOURCE_PATH_FINAL, 'train_anno_pic.csv')

# 处理后的数据目录
PROCESSED_PATH = os.path.join(DATA_PATH, 'processed')
# 处理后的IMG数据目录
PROCESSED_TEST_IMG_PATH = os.path.join(PROCESSED_PATH, 'img', 'test')
PROCESSED_TRAIN_IMG_PATH = os.path.join(PROCESSED_PATH, 'img', 'train')
# 处理后的CSV数据目录
PROCESSED_CSV_PATH = os.path.join(PROCESSED_PATH, 'csv')
os.makedirs(PROCESSED_TEST_IMG_PATH, exist_ok=True)
os.makedirs(PROCESSED_TRAIN_IMG_PATH, exist_ok=True)
os.makedirs(PROCESSED_CSV_PATH, exist_ok=True)

PROCESSED_TRAIN_CSV_PATH = os.path.join(PROCESSED_CSV_PATH, 'train.csv')
PROCESSED_TEST_CSV_PATH = os.path.join(PROCESSED_CSV_PATH, 'test.csv')
PROCESSED_IMAGE_CSV_PATH = os.path.join(PROCESSED_CSV_PATH, 'image_data.csv')

# 模型文件路径
MODEL_FILE_PATH = os.path.join(root_path, 'model', 'ckpt')
os.makedirs(MODEL_FILE_PATH, exist_ok=True)

# 预测结果路径
PREDICT_RESULT_PATH = os.path.join(root_path, 'model', 'result')
os.makedirs(PREDICT_RESULT_PATH, exist_ok=True)
