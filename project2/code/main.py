# 获取当前脚本目录
from recognize import recognize
from segment_img import split_car_plate
import os
import sys


current_path = os.path.dirname(os.path.abspath(__file__))
# 添加到 sys.path
sys.path.append(current_path)


if __name__ == "__main__":
    input_img = "../imgs/input.jpg"
    char_img_paths = split_car_plate(input_img)
    result = [recognize(path) for path in char_img_paths]
    print(result)
