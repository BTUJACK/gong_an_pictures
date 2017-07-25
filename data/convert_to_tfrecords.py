# -*- coding: utf-8 -*-
__author__ = 'oukohou'
__time__ = '17-6-6 上午9:06'

"""
If this runs wrong, don't ask me, I don't know why;
If this runs right, thank god, and I don't know why.
Maybe the answer, my friend, is blowing in the wind.
"""
import argparse
import os
import sys
import time

import tensorflow as tf
from PIL import Image


def create_record(class_path, resize_height, resize_width, depth):
    current_time = time.strftime("_%Y-%m-%d_%H-%M-%S", time.localtime())
    path_to_write = "./tf_records/" + os.path.basename(class_path) + current_time + ".tfrecords"
    writer = tf.python_io.TFRecordWriter(path_to_write)
    for path, subdirs, files in os.walk(class_path):
        for filename in files:
            image_path = os.path.join(path, filename)
            label = int(path.split(os.sep)[-1])
            img = Image.open(image_path).resize((resize_height, resize_width))
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features
                (feature={
                "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[resize_height])),
                "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[resize_width])),
                "depth": tf.train.Feature(int64_list=tf.train.Int64List(value=[depth])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }
            ))
            writer.write(example.SerializeToString())
    writer.close()


def main(_):
    create_record(Flags.class_path, Flags.height, Flags.width, 3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert raw images to tfrecord format...")
    parser.add_argument(
        '--class_path', type=str,
        # default="/home/oukohou/study/tensorflow/oukohou/gong_an_pictures/data/train/1000_per_gender",
        default="./train/gender_for_age_from_93_to_2000",
        help='directory that contains the subfolders of images.')
    parser.add_argument(
        '--height', type=int, default=766,
        help='height you want to resize the image to.')
    parser.add_argument(
        '--width', type=int, default=574,
        help='width you want to resize the image to')
    Flags, unparsed = parser.parse_known_args()
    # tf.app.run(main=main)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    args = parser.parse_args()
    args.log.close()
