
import numpy as np
import cv2
import os


def read_and_convert_to_binary(image_path, output_path):
    # 加载图像并直接转换为灰度图像
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 应用阈值，将灰度图像转换为二值图像
    _, binary_image = cv2.threshold(
        gray_image, 0, 255,  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    resize = cv2.resize(binary_image, (40, 20),
                        interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(output_path, resize)

    return binary_image


def process_images_in_directory(src_root, dst_root):
    # 遍历源目录
    for subdir, _, files in os.walk(src_root):
        for file in files:
            # 构建源文件路径
            src_file_path = os.path.join(subdir, file)
            # 构建目标文件路径
            relative_path = os.path.relpath(src_file_path, src_root)
            dst_file_path = os.path.join(dst_root, relative_path)

            # 确保目标目录存在
            os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
            read_and_convert_to_binary(src_file_path, dst_file_path)


# 定义源目录和目标目录

# 调用函数处理图片
if __name__ == "__main__":
    src_root = "../template/refer1"
    dst_root = "../template/refer2"
    process_images_in_directory(src_root, dst_root)
