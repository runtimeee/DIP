import json
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# 加载已经训练好的模型
model = tf.keras.models.load_model("../model.h5")

# 定义预处理图像函数


def preprocess_image(img_path, target_size=(40, 20)):
    # 加载图像
    img = image.load_img(img_path, target_size=target_size)
    # 将图像转换为numpy数组
    img_array = image.img_to_array(img)
    # 增加一个维度，使其成为 (1, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    # 归一化图像
    img_array /= 255.0
    return img_array


def predict_image(model, img_array, class_indices):
    # 使用模型进行预测
    predictions = model.predict(img_array)
    # 找到概率最高的类别
    predicted_class_index = np.argmax(predictions[0])
    # 获取类别名称
    class_names = list(class_indices.keys())
    predicted_class = class_names[predicted_class_index]
    return predicted_class, predictions[0]


class_indices = None
with open('class_indices.json', 'r', encoding='utf-8') as f:
    class_indices = json.load(f)


def recognize(char_img_path):
    img_array = preprocess_image(char_img_path)
    # 进行预测
    predicted_class, _ = predict_image(
        model, img_array, class_indices)
    return str(predicted_class)


if __name__ == "__main__":
    for i in range(7):
        # 预处理图像
        img_path = f"../imgs/{i}c.jpg"
        img_array = preprocess_image(img_path)
        # 进行预测
        predicted_class, prediction_probabilities = predict_image(
            model, img_array, class_indices)

        print(f"Predicted class: {predicted_class}")
