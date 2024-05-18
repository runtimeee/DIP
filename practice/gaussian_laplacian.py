import cv2
import numpy as np



import cv2
import numpy as np

def region_filling(image_path, output_path):

    # 读取图像
    img = cv2.imread(input_path)
    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二值化处理
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    size = int(2 * (np.ceil(3 * sigma)) + 1)
    log_filter = cv2.getGaussianKernel(size, sigma) * cv2.getGaussianKernel(size, sigma).T
    log_filter = log_filter * (np.square(np.arange(size) - size // 2)[:, None] + np.square(np.arange(size) - size // 2)[None, :])
    log_filter = log_filter - log_filter.mean()

    # 应用高斯-拉普拉斯滤波器
    log_image = cv2.filter2D(image, cv2.CV_64F, log_filter)
    
    # 保存处理后的图像
    cv2.imwrite(output_path, log_image)



if __name__ == '__main__':
    input_path = '/workspace/DIP/project1/src_imgs/hit.png'
    output_path = '/workspace/DIP/practice/tmp/morph.png'
    # boundary_extraction(input_path, output_path)

    input_path = '/workspace/DIP/project1/src_imgs/image_pro_1_2.jpg'
    # input_path = '/workspace/DIP/project1/src_imgs/hit.png'
    output_path = '/workspace/DIP/practice/tmp/morph1.jpg'
    region_filling(input_path, output_path)

