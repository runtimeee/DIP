import cv2
import numpy as np

def median_filter(img, ksize=3):
    """
    对图像进行中值滤波
    
    参数:
        img (numpy.ndarray): 输入图像
        ksize (int): 滤波器大小,必须为奇数
    
    返回:
        numpy.ndarray: 去噪后的图像
    """
    rows, cols = img.shape[:2]
    new_img = np.zeros_like(img)
    
    # 填充图像边界
    padded = cv2.copyMakeBorder(img, ksize//2, ksize//2, ksize//2, ksize//2, cv2.BORDER_REPLICATE)
    
    for i in range(rows):
        for j in range(cols):
            # 获取当前像素周围的ksize x ksize邻域
            neighborhood = padded[i:i+ksize, j:j+ksize]
            
            # 计算邻域中像素值的中值
            median = np.median(neighborhood)
            
            # 将中值赋给输出图像对应位置
            new_img[i, j] = median

    return new_img

def denoise_image(input_path, output_path, ksize=3):
    """
    使用中值滤波对含有椒盐噪声的图像进行去噪处理
    
    参数:
        input_path (str): 输入图像路径
        output_path (str): 输出去噪图像路径
        ksize (int): 滤波器大小,必须为奇数
    """
    
    # 读取输入图像并调整大小
    img = cv2.imread(input_path)
    img = cv2.resize(img, (440, 280))
    
    # 对图像进行中值滤波去噪
    denoised_img = median_filter(img, ksize)
    
    # 保存去噪图像
    cv2.imwrite(output_path, denoised_img)


if __name__ == '__main__':
    input_path = 'src_imgs/hit.png'
    output_path = 'result_1_1.png'
    denoise_image(input_path, output_path)