#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@version: python3.7
@author: ‘changzhaoliang‘
@license: Apache Licence 
@file: image.py
@time: 2019-04-17 11:30
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf


# 读取数据
def get_decode_image(file_name, dtype=tf.uint8, has_eval=False):
    # 读取原始数据
    image_raw_data = tf.gfile.FastGFile(file_name, "rb").read()

    # 解码图片数据得到三维矩阵
    # image_data = tf.image.decode_jpeg(image_raw_data)
    # image_data = tf.image.decode_png(image_raw_data)
    image_data = tf.image.decode_image(image_raw_data)

    # 转换数据类型
    image_data = tf.image.convert_image_dtype(image_data, dtype=dtype)

    if has_eval:
        with tf.Session():
            image_data = image_data.eval()

    return image_data


# 由于图片存在压缩，将一张图像还原成三维矩阵，需要进行解码。
def decode_image_and_save(file_name="data_process.jpg"):
    name, ext = os.path.splitext(file_name)
    # 读取数据
    image_data = get_decode_image(file_name)

    # 编码图片
    encoded_image = tf.image.encode_jpeg(image_data) if ext.lower() == "png" else tf.image.encode_png(image_data)
    ext = "jpg" if ext.lower() == "png" else "png"

    with tf.Session():
        # 保存图片
        with tf.gfile.GFile(name="{}.{}".format(name, ext), mode="wb") as f:
            f.write(encoded_image.eval())

        plt.imshow(image_data.eval())
        plt.show()
    pass


# 图片大小调整:通过算法使新的图像尽量保存原始图像的所有信息。
def resize_image(file_name="data_process.jpg"):
    # 读取数据
    image_data = get_decode_image(file_name, dtype=tf.float32, has_eval=True)

    # 1.调整大小：插值
    resized_images = tf.image.resize_images(image_data, [300, 300], method=tf.image.ResizeMethod.BILINEAR)
    # 2.调整大小：crop or pad
    # resized_images = tf.image.resize_image_with_crop_or_pad(image_data, target_height=500, target_width=500)
    # 3.调整大小：比例
    # resized_images = tf.image.central_crop(image_data, 0.5)
    # 4.调整大小：裁剪
    # resized_images = tf.image.crop_to_bounding_box(image_data, 100, 100, target_height=400, target_width=500)
    # 5.调整大小：填充
    # resized_images = tf.image.pad_to_bounding_box(image_data, 400, 100, target_height=1000, target_width=1200)
    # 6.调整大小：裁剪并resize
    # image_data = tf.expand_dims(image_data, axis=0)
    # resized_images = tf.image.crop_and_resize(image_data, [[0.8, 0.2, 0.5, 0.5]], box_ind=[0], crop_size=[200, 200])
    # resized_images = tf.squeeze(resized_images, axis=0)

    print(resized_images.get_shape())

    # 显示图片
    with tf.Session():
        plt.imshow(resized_images.eval())
        plt.show()
    pass


# 图像翻转
def flip_image(file_name="data_process.jpg"):
    image_data = get_decode_image(file_name, dtype=tf.float32)

    # 翻转
    # flip_image_data = tf.image.flip_up_down(image_data)
    # flip_image_data = tf.image.flip_left_right(image_data)
    # 随机翻转
    # flip_image_data = tf.image.random_flip_left_right(image_data)
    # flip_image_data = tf.image.random_flip_up_down(image_data)
    # 转置
    flip_image_data = tf.image.transpose_image(image_data)

    print(flip_image_data.get_shape())

    with tf.Session():
        plt.imshow(flip_image_data.eval())
        plt.show()
    pass


# 图像色彩调整:亮度、对比度、色相、饱和度
def adjust_color(file_name="data_process.jpg"):
    image_data = get_decode_image(file_name, dtype=tf.uint8, has_eval=True)

    # 亮度
    adjust_image_data = tf.image.adjust_brightness(image_data, 0.5)
    # adjust_image_data = tf.image.random_brightness(image_data, 0.5)

    # 对比度:(x - mean) * contrast_factor + mean
    # adjust_image_data = tf.image.adjust_contrast(image_data, 5)
    # adjust_image_data = tf.image.adjust_contrast(image_data, -5)
    # adjust_image_data = tf.image.random_contrast(image_data, 2, 5)

    # 色相
    # adjust_image_data = tf.image.adjust_hue(image_data, 0.5)
    # adjust_image_data = tf.image.adjust_hue(image_data, -0.5)
    # adjust_image_data = tf.image.random_hue(image_data, 0.4)

    # 饱和度
    # adjust_image_data = tf.image.adjust_saturation(image_data, 5)
    # adjust_image_data = tf.image.adjust_saturation(image_data, 0.4)
    # adjust_image_data = tf.image.random_saturation(image_data, 0.4, 10)

    # 标准化:(x - mean) / adjusted_stddev
    # adjust_image_data = tf.image.per_image_standardization(image_data)

    with tf.Session():
        plt.imshow(adjust_image_data.eval())
        plt.show()
    pass


# 标准化
def standard_image(file_name="data_process.jpg"):
    image_data = get_decode_image(file_name, dtype=tf.uint8, has_eval=True)

    # 标准化:(x - mean) / adjusted_stddev
    standard_image_data = tf.image.per_image_standardization(image_data)

    with tf.Session():
        print(standard_image_data.eval())
    pass


# 处理标注框
def draw_bounding_boxes(file_name="data_process.jpg"):
    image_data = get_decode_image(file_name, dtype=tf.float32, has_eval=True)
    image_data = tf.image.resize_images(image_data, [500, 500], method=tf.image.ResizeMethod.BILINEAR)
    image_data = tf.expand_dims(image_data, axis=0)
    result = tf.image.draw_bounding_boxes(image_data, boxes=[[[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.5, 0.5]]])
    result = tf.squeeze(result, axis=0)
    with tf.Session():
        plt.imshow(result.eval())
        plt.show()
    pass


def image_read(file_name):
    image_jpg = tf.gfile.FastGFile(file_name, 'rb').read()
    with tf.Session() as sess:
        image_jpg = tf.image.decode_jpeg(image_jpg)  # 图像解码

        print(sess.run(image_jpg))  # 打印解码后的图像（即为一个三维矩阵[w,h,3]）
        image_jpg = tf.image.convert_image_dtype(image_jpg, dtype=tf.float32)  # 改变图像数据类型
        print(sess.run(image_jpg))

        plt.figure(1)  # 图像显示
        plt.imshow(image_jpg.eval())
        plt.show()



if __name__ == '__main__':
    # draw_bounding_boxes()
    image_read('./image/test.jpg')