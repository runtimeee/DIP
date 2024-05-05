import cv2
import numpy as np


def dfs(image, x, y, visited, contour, start_x, start_y, prev_x, prev_y):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右四个方向
    stack = [(x, y, prev_x, prev_y)]
    while stack:
        cx, cy, px, py = stack.pop()
        if visited[cx][cy]:
            if (cx, cy) != (start_x, start_y) or len(contour) > 3:
                continue
        visited[cx][cy] = True
        contour.append((cx, cy))
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1]:
                if image[nx][ny] > 0 and (nx, ny) != (px, py):
                    stack.append((nx, ny, cx, cy))

def find_contours(image):
    """
    通过dfs找到图像中所有的闭环轮廓点集合
    
    参数:
        image: 输入图像
    返回:
        处理后的图像
    """
    rows, cols = image.shape
    visited = np.zeros((rows, cols), dtype=bool)
    contours = []
    
    for i in range(rows):
        for j in range(cols):
            if image[i][j] > 0 and not visited[i][j]:
                contour = []
                dfs(image, i, j, visited, contour, i, j, -1, -1)
                if len(contour) >= 3:  # 简单的过滤，确保轮廓是闭环
                    contours.append(contour)
    return contours




def is_point_in_polygon(x, y, polygon):
    """
    判断点是否在多边形内部
    """
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n+1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def traverse_polygon(image, polygon):
    """
    遍历多边形内部所有的点，并设置颜色为白色
    """
    min_x = min([point[0] for point in polygon])
    max_x = max([point[0] for point in polygon])
    min_y = min([point[1] for point in polygon])
    max_y = max([point[1] for point in polygon])
    
    # 遍历多边形内的所有点
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            if is_point_in_polygon(x, y, polygon):
                image[x][y] = 255

def draw_contours(image, contours):
    """
    使用白色填充闭合的轮廓
    """
    for contour in contours:
        traverse_polygon(image, contour)
    return image



def post_process_by_morphology(binary, kernel_size=(5, 5), iterations=3):
    """
    再次使用形态学闭运算、连接断裂和填补小孔
    
    参数:
        binary: 输入文本图像路径
        kernel_size: 闭运算使用的结构元素大小,默认为(5,5)
        iterations: 闭运算的迭代次数,默认为3
    返回:
        处理后的图像
    """

    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    # 执行闭运算
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    # 对原始图像使用闭运算结果作为掩码
    result = cv2.bitwise_and(binary, binary, mask=closed)

    return result

def fill_text_holes_by_contours_by_self(input_path, output_path,):
    """
    使用轮廓绘制来填充文本图像中的字符孔洞
    
    参数:
        input_path: 输入文本图像路径
        output_path: 输出填充后图像的路径
    """
    # 读取图像
    img = cv2.imread(input_path)
    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化处理
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
  
    # 找到轮廓
    contours = find_contours(binary)
    draw_contours(binary, contours)

    binary = post_process_by_morphology(binary)
    
    # 保存处理后的图像
    cv2.imwrite(output_path, binary)

def fill_text_holes_by_contours_cv2(input_path, output_path,):
    """
    使用轮廓绘制来填充文本图像中的字符孔洞
    
    参数:
        input_path: 输入文本图像路径
        output_path: 输出填充后图像的路径
    """
    # 读取图像
    img = cv2.imread(input_path)
    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化处理
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 找到轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 绘制轮廓来填充孔洞
    cv2.drawContours(binary, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # 
    binary = post_process_by_morphology(binary)


    # 保存处理后的图像
    cv2.imwrite(output_path, binary)






if __name__ == '__main__':
    input_path = 'src_imgs/image_pro_1_2.jpg'
    output_path = 'result_2.jpg'
    # fill_text_holes_by_contours_by_self(input_path, output_path)
    fill_text_holes_by_contours_cv2(input_path, output_path)

