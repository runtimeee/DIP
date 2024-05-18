import cv2
import numpy as np

def boundary_extraction(image_path, output_path):
    # 读取图像
    img = cv2.imread(image_path, 0)  # 0 表示以灰度模式读取
    if img is None:
        print("图像文件读取失败，请检查路径是否正确。")
        return

    # 应用阈值化处理将图像转换为二值图像
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 使用3x3的结构元素
    kernel = np.ones((3,3), np.uint8)

    # 腐蚀操作
    eroded_img = cv2.erode(binary_img, kernel, iterations=1)

    # 边界提取：原图像减去腐蚀后的图像
    boundary = cv2.subtract(binary_img, eroded_img)


    # 保存处理后的图像
    cv2.imwrite(output_path, boundary)


import cv2
import numpy as np

def region_filling(image_path, output_path):

    # 读取图像
    img = cv2.imread(input_path)
    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二值化处理
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    cv2.imwrite(output_path.split(".")[0]+"_bin.png", binary)

    # kernel = np.ones((3,3), np.uint8)
    kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)

    # new_binary = cv2.dilate(binary, kernel, iterations=1)
    new_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(output_path.split(".")[0]+"_new_bin.png", new_binary)

    # 获取图像补集
    img_complement = cv2.bitwise_not(new_binary)

    # 初始化填充区域，X0为种子点
    h, w = new_binary.shape
    img_floodfill = np.zeros((h, w), np.uint8)
    img_floodfill[h//2,w//2] = 255

    # 定义结构元素B

    while True:
        # Xk-1 膨胀操作
        img_dilated = cv2.dilate(img_floodfill, kernel, iterations=1)
        # (Xk-1 ⊕ B) ∩ Ac
        img_floodfill_new = cv2.bitwise_and(img_dilated, img_complement)

        # 检查是否有变化，如果Xk = Xk-1，则停止迭代
        if np.array_equal(img_floodfill, img_floodfill_new):
            break

        img_floodfill = img_floodfill_new.copy()
    
    cv2.imwrite(output_path, img_floodfill)
    





if __name__ == '__main__':
    input_path = '/workspace/DIP/project1/src_imgs/hit.png'
    output_path = '/workspace/DIP/practice/tmp/morph.png'
    # boundary_extraction(input_path, output_path)

    input_path = '/workspace/DIP/project1/src_imgs/image_pro_1_2.jpg'
    # input_path = '/workspace/DIP/project1/src_imgs/hit.png'
    output_path = '/workspace/DIP/practice/tmp/morph1.jpg'
    region_filling(input_path, output_path)

