import os

import cv2


class Processing:

    def blur_canny_cst(self, img_path):
        """
        CST图像处理
        :param img_path:
        :return:
        """
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # 降噪
        # img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.medianBlur(img, 5)

        # 边缘化
        img = cv2.Canny(img, 32, 256)

        return img
