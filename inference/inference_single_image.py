# -*- coding: utf-8 -*-
__author__ = 'oukohou'
__time__ = '2017/7/6 14:47'

# If this runs wrong, don't ask me, I don't know why;
# If this runs right, thank god, and I don't know why.
# Maybe the answer, my friend, is blowing in the wind.

import sys

import tensorflow as tf
from PIL import Image
import numpy as np

from train.CNN_for_tfrecords import Flags

height = Flags.height
width = Flags.width
IMAGE_PIXELS = height * width * 3  # default


def read_single_image(image_path):
    # image = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_UNCHANGED)
    image = Image.open(image_path)
    image = tf.convert_to_tensor(np.asarray(image))
    # image =tf.contrib.keras.preprocessing.image.load_img(image_path)# [image_path]
    # image_queue = tf.train.string_input_producer(image)
    # reader = tf.WholeFileReader()
    # key , value = reader.read(image_queue)
    # image = tf.image.decode_jpeg(value,channels=3)
    assert image is not None
    image = tf.image.resize_image_with_crop_or_pad(
        image=image,
        target_height=height,
        target_width=width,
    )
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    image = tf.reshape(image, [-1, height, width, 3])
    return image


def CNN_model(class_num, input_image):
    input_layer = tf.reshape(input_image, [-1, height, width, 3], name="input_layer")

    conv_before_1 = tf.layers.conv2d(input_layer, filters=64, kernel_size=[3, 3], activation=tf.nn.relu,
                                     name="conv_before_1")
    pool_before_1 = tf.layers.max_pooling2d(conv_before_1, pool_size=[2, 2], strides=[2, 2], name="pool_before_1")
    conv_before_2 = tf.layers.conv2d(pool_before_1, filters=256, kernel_size=[3, 3], activation=tf.nn.relu,
                                     name="conv_berfore_2")
    pool_before_2 = tf.layers.max_pooling2d(conv_before_2, pool_size=[2, 2], strides=[2, 2], name="pool_before_2")

    pool0 = tf.layers.max_pooling2d(pool_before_2, pool_size=[2, 2], strides=[2, 2], name="pool0")
    conv1 = tf.layers.conv2d(pool0, filters=512, kernel_size=[3, 3], strides=(1, 1),
                             padding="valid", activation=tf.nn.relu, name="conv1")
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=[2, 2], name="pool1")
    conv2 = tf.layers.conv2d(pool1, filters=256, kernel_size=[3, 3],
                             padding="valid", activation=tf.nn.relu, name="conv2")
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=[2, 2], name="pool2")
    conv3 = tf.layers.conv2d(pool2, filters=144, kernel_size=[3, 3], padding="valid",
                             activation=tf.nn.relu, name="conv3")
    pool3 = tf.layers.max_pooling2d(conv3, pool_size=[2, 2], strides=[2, 2], name="pool3")
    pool3_flatten = tf.reshape(pool3, [-1, 7 * 10 * 48], name="pool3_flatten")
    dense = tf.layers.dense(pool3_flatten, units=1024, activation=tf.nn.relu, name="dense")
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, seed=1020, training=True, name="dropout")
    logits = tf.layers.dense(dropout, units=class_num, activation=tf.nn.relu, name="logits")  # [batch_size, class_num]
    return logits


# s = ""
# s.end

def predict_single_image(image_path, train_log_path):
    # if sys.argv[0].endswith(""):
    checkpoint = tf.train.get_checkpoint_state(train_log_path)
    input_checkpoint = checkpoint.model_checkpoint_path

    image = read_single_image(image_path)
    logits = CNN_model(Flags.class_num, image)

    output_score = tf.nn.softmax(logits)
    predict_value, predict_index = tf.nn.top_k(output_score, k=2)
    label_names = get_label_names()

    saver = tf.train.Saver()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(init_op)
        saver.restore(sess, input_checkpoint)
        predict_value, predict_index = sess.run([predict_value, predict_index])
        for i in range(len(predict_index[0])):
            print('[INFO] predict as: %s,  probabilities: %.4f' %
                  (label_names[predict_index[0][i]], predict_value[0][i]))


def get_label_names(gender=True):
    label_names = {}
    if gender:  # to judge a photo as female or male.
        label_names[0] = "female"
        label_names[1] = "male"

    else:  # to judge a photo's age, which is now not supported.
        # label_names;
        pass
    return label_names


image_path_ = sys.argv[1]
# image_path_ = "E:\gong_an_pictures0612\gong_an_pictures\data\\test_data\\500_per_gender\\0\9900.JPG"
# print(image_path_)
# train_log_path = "../train_log"
train_log_path = sys.argv[2]
predict_single_image(image_path_, train_log_path)
