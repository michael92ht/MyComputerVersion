# -*- coding: utf-8 -*-
# 对目标识别问题中的图像数据进行增强，核心点为：将初始图像按照一定的角度进行旋转，得到新的图像同时，并得到新的标注点。

from __future__ import division
import cv2
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool


# 计算图像按照angle角度进行旋转后的仿射变换矩阵
def get_rotation_mat(image, angle, ratio=0.75):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_mat = cv2.getRotationMatrix2D(center, angle, ratio)
    return rotation_mat


# 将仿射变换矩阵应用在图像上
def get_rotated_image(image, rotation_mat):
    (h, w) = image.shape[:2]
    return cv2.warpAffine(image, rotation_mat, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# 对一个坐标点，得到变换后的坐标点
def get_rotated_points(point, rotation_mat):
    pos = np.mat([[point[0]], [point[1]], [1]])
    rotated_pos = np.array(np.mat(rotation_mat) * pos)
    return int(rotated_pos[0][0]), int(rotated_pos[1][0])


# 获取水表图像的度数与读数框位置
def get_img_details(image_path):
    basename = os.path.basename(image_path)
    temp = basename.split('_')
    points = temp[-1][:-4].split('-')
    points = [int(p) for p in points]
    assert len(points) == 8, "The length of points should be 8."
    for p in points:
        assert 1000 >= p >= 0, "Points should be in range[0, 1000]"
    numbers = temp[-2]
    assert len(numbers) == 5, "The length of numbers should be 5."
    return points, numbers


# 获取对角线坐标点
def diagonal_points(points):
    xmin = min(points[::2])
    xmax = max(points[::2])
    ymin = min(points[1::2])
    ymax = max(points[1::2])
    return xmin, ymin, xmax, ymax


def to_4_points(points):
    assert len(points) == 8, "The length of points to 4 should be 8."
    return [(points[0], points[1]),
            (points[2], points[3]),
            (points[4], points[5]),
            (points[6], points[7])]


def to_8_points(points):
    assert len(points) == 4, "The length of points to 8 should be 4."
    eight_points = list()
    [eight_points.extend([x[0], x[1]]) for x in points]
    return eight_points


def rotate_image(inputs):
    image, dst = inputs
    print("Processing: ", image)
    basename = os.path.basename(image)
    name = basename.split('_')[:-2]
    points, numbers = get_img_details(image)
    points = to_4_points(points)
    img = cv2.imread(image)
    for angle in range(10, 351, 10):
        mat = get_rotation_mat(img, angle, 0.8)
        rimg = get_rotated_image(img, mat)
        rpoints = [get_rotated_points(p, mat) for p in points]
        eight_points = to_8_points(rpoints)
        for rp in eight_points:
            assert rp in range(0, 1000)
        temp = list()
        temp.extend(name)
        temp.extend([str(angle).zfill(5), numbers,
                     "-".join([str(p) for p in eight_points])])
        newname = "_".join(temp)
        write_name = os.path.join(dst, newname + '.jpg')
        cv2.imwrite(write_name, rimg)
        print("Rotated image write to", write_name)


# 旋转图像
def rotate_imgs(src, dst):
    assert os.path.exists(src), "Images path is not exists." + src
    if not os.path.exists(dst):
        os.mkdir(dst)
    images = [os.path.join(src, x) for x in os.listdir(src) if x.endswith('.jpg')]
    inputs = [(image, dst) for image in images]
    pool = ThreadPool(30)
    pool.map(rotate_image, inputs)
    pool.close()
    pool.join()


def imgs_augmentation(src, dst):
    assert os.path.exists(src), "Images path is not exists." + src
    if not os.path.exists(dst):
        os.mkdir(dst)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    images = [os.path.join(src, x) for x in os.listdir(src) if x.endswith('.jpg')]
    for image in images:
        print("Processing: ", image)

        image_raw_data = tf.gfile.FastGFile(image, 'rb').read()
        img_data = tf.image.decode_jpeg(image_raw_data)

        # 计算图像的亮度均值，对亮度值较大或较小区别处理

        gray = tf.image.rgb_to_grayscale(img_data)
        brightness_mean = np.mean(gray.eval())

        # 随机变换图像
        # 亮度
        if 100 > brightness_mean > 50:
            brightness_delta = 0.3
        elif 130 > brightness_mean >= 100:
            brightness_delta = 0.1
        else:
            brightness_delta = 0.05

        adjusted = tf.image.random_brightness(img_data, max_delta=brightness_delta)

        # 色调
        adjusted = tf.image.random_hue(adjusted, max_delta=0.2)
        # 饱和度
        adjusted = tf.image.random_saturation(adjusted, 0, 3)

        modified_img = Image.fromarray(adjusted.eval())

        basename = os.path.basename(image)
        write_name = os.path.join(dst, basename)

        modified_img.save(write_name)
        print("Rotated image write to", write_name)


# 获取读数框小图像
def crop_imgs(src, dst):
    assert os.path.exists(src), "Images path is not exists." + src
    if not os.path.exists(dst):
        os.mkdir(dst)
    images = [os.path.join(src, x) for x in os.listdir(src) if x.endswith('.jpg')]
    for image in images:
        print("Processing: ", image)
        basename = os.path.basename(image)
        points, _ = get_img_details(image)
        xmin, ymin, xmax, ymax = diagonal_points(points)
        img = cv2.imread(image)
        crop_img = img[ymin: ymax, xmin: xmax]
        write_name = os.path.join(dst, basename)
        cv2.imwrite(write_name, crop_img)
        print("Rotated image write to", write_name)


# 转化为OCR训练所需图像
def ocr_imgs(src, dst):
    assert os.path.exists(src), "Images path is not exists." + src
    if not os.path.exists(dst):
        os.mkdir(dst)
    images = [os.path.join(src, x) for x in os.listdir(src) if x.endswith('.jpg')]
    for image in images:
        print("Processing: ", image)
        basename = os.path.basename(image)
        img = Image.open(image)
        w, h = img.size
        if h > w:
            img = img.transpose(Image.ROTATE_90)
        left = img.resize((200, 100))
        right = img.transpose(Image.ROTATE_180).resize((200, 100))
        merge_img = np.hstack((left, right))
        write_name = os.path.join(dst, basename)
        cv2.imwrite(write_name, merge_img)
        print("Rotated image write to", write_name)


if __name__ == "__main__":
    src = r"/bigdata/datasets/images/image_8135"
    rotate = r"/bigdata/datasets/roated_8135"
    augment = r"/bigdata/datasets/augment_8135"
    sub = r"/bigdata/datasets/sub_8135"
    ocr = r"/bigdata/datasets/ocr_8135"
    rotate_imgs(src, rotate)
    imgs_augmentation(rotate, augment)
    crop_imgs(augment, sub)
    ocr_imgs(sub, ocr)
    # imgs_augmentation(dst, ten)
    pass

    # src = r"/bigdata/datasets/images/image_581"
    # rotate = r"/bigdata/datasets/roated_581"
    # augment = r"/bigdata/datasets/augment_581"
    # sub = r"/bigdata/datasets/sub_581"
    # ocr = r"/bigdata/datasets/ocr_581"
    # rotate_imgs(src, rotate)
    # imgs_augmentation(rotate, augment)
    # crop_imgs(augment, sub)
    # ocr_imgs(sub, ocr)
    # imgs_augmentation(dst, ten)
    pass

