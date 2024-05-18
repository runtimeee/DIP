import cv2
import numpy as np


import cv2
import numpy as np
from scipy.signal import convolve2d

def sobel_edge_detection_manual(image_path, output_path, threshold=100):
    """
    使用sobel算子检测图像边缘
    
    参数:
        input_path (str): 输入图像路径
        output_path (str): 输出图像路径
        threshold (int): 门限值
    """
    # 使用cv2读取图像并转换为灰度
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 定义Sobel算子
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])
    
    # 使用卷积函数应用Sobel算子
    grad_x = convolve2d(img, sobel_x, mode='same', boundary='fill', fillvalue=0)
    grad_y = convolve2d(img, sobel_y, mode='same', boundary='fill', fillvalue=0)
    
    # 计算梯度幅度
    sobel = np.sqrt(grad_x**2 + grad_y**2)
    
    # 应用阈值来获取边缘图像
    edge_image = np.where(sobel > threshold, 255, 0).astype(np.uint8)

    # 使用cv2.imwrite将边缘图像写入到输出路径
    cv2.imwrite(output_path, edge_image)


    # 
    # 图像相减
    result = cv2.subtract(img, edge_image)

    # 保存结果
    cv2.imwrite('sub.png', result)



if __name__ == '__main__':
    input_path = 'result_1_1.png'
    output_path = 'result_1_2.png'
    sobel_edge_detection_manual(input_path, output_path, 100)