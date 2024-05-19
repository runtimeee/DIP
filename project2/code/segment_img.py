import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_image(image_path):
    """读取图像"""
    image = cv2.imread(image_path)
    return image


def convert_to_hsv(image):
    """将图像转换为HSV颜色空间"""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image


def filter_blue(image):
    """颜色过滤，提取蓝色区域"""
    hsv = convert_to_hsv(image)
    # 蓝色的HSV范围
    lower_blue = np.array([100, 80, 80])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    return mask


def filter_white(image):
    """颜色过滤，提取白色区域"""
    hsv = convert_to_hsv(image)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 20, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    return mask_white


def morphological_operations(mask, k=5):
    """形态学操作，去除噪声"""
    kernel = np.ones((k, k), np.uint8)
    morphed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel)
    return morphed


def find_plate_contour(mask, image):
    """找到车牌的轮廓并绘制轮廓"""
    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 2 < w / h < 5 and w > 50 and h > 20:  # 根据车牌的宽高比和尺寸进行筛选
            plate_image = image[y:y+h, x:x+w]
            return plate_image
    return None


def preprocess_plate_image(plate_image):
    """车牌图像预处理"""
    gray_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    m1 = morphological_operations(gray_plate, k=3)
    _, binary_plate = cv2.threshold(
        m1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_plate


def segment_characters(binary_image):
    # 查找连通组件
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8)

    # 提取字符
    characters = []
    for i in range(1, num_labels):  # 跳过背景
        x, y, w, h, area = stats[i]
        if area < 50:
            continue
        character = binary_image[y:y+h, x:x+w]
        characters.append((x, character))

    characters.sort(key=lambda i: i[0])
    return [x[1] for x in characters]


def segment_characters_(binary_plate):
    """分割出字符"""
    contours, _ = cv2.findContours(
        binary_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    character_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h / binary_plate.shape[0] > 0.5:  # 根据字符的高度进行筛选
            character = binary_plate[y:y+h, x:x+w]
            character_images.append(character)
    return character_images


def split_car_plate(image_path):
    # 读取图像
    image = read_image(image_path)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 颜色过滤，提取蓝色区域
    mask = filter_blue(image)
    cv2.imwrite('../imgs/blue.jpg', mask)

    # 颜色过滤，提取白色边框
    mask_white = filter_white(image)
    cv2.imwrite('../imgs/white.jpg', mask_white)

    # 结合蓝色和白色的掩码，去除白色边框但保留白色字符
    mask_combined = cv2.bitwise_and(mask, cv2.bitwise_not(mask_white))

    cv2.imwrite('../imgs/combined.jpg', mask_combined)

    # 形态学操作，去除噪声
    morphed = morphological_operations(mask_combined)
    cv2.imwrite('../imgs/morphed.jpg', mask)

    # 找到车牌的轮廓并绘制轮廓
    plate_image = find_plate_contour(morphed, image)
    if plate_image is None:
        print("failed to find car plate")

    cv2.imwrite('../imgs/car_plate.jpg', plate_image)
    # 车牌图像预处理
    binary_plate = preprocess_plate_image(plate_image)
    cv2.imwrite('../imgs/car_plate_post.jpg', binary_plate)

    characters = segment_characters(binary_plate)
    img_path_list = []
    for i, character in enumerate(characters):
        img_path = f'../imgs/{i}c.jpg'
        cv2.imwrite(img_path, character)
        img_path_list.append(img_path)

    return img_path_list


if __name__ == "__main__":
    # 示例调用
    image_path = '../imgs/input.jpg'
    split_car_plate(image_path)
