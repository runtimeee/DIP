import json
import os
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def prepare_data(src_root, img_height=224, img_width=224, batch_size=32, test_size=0.2):
    data_gen = ImageDataGenerator(
        rescale=1.0/255.0, validation_split=test_size)

    train_gen = data_gen.flow_from_directory(
        src_root,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_gen = data_gen.flow_from_directory(
        src_root,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_gen, val_gen


def build_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# 定义路径和参数
src_root = "template/refer2"
img_height, img_width = 40, 20
batch_size = 32
test_size = 0.2

# 准备数据集
train_gen, val_gen = prepare_data(
    src_root, img_height, img_width, batch_size, test_size)

with open('class_indices.json', 'w') as f:
    json.dump(train_gen.class_indices, f, ensure_ascii=False, indent=4)


# 构建模型
num_classes = len(train_gen.class_indices)
input_shape = (img_height, img_width, 3)
model = build_model(input_shape, num_classes)

# 训练模型
epochs = 3
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs
)

# 保存模型
model.save("model.h5")
