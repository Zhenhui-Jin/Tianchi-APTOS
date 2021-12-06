from processing import data_factory


def processing_data():
    """
    预处理数据
    :return:
    """
    data_factory.crop_img()
    data_factory.processing_data_source_preliminary()


if __name__ == '__main__':
    processing_data()
