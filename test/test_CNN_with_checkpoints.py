# -*- coding: utf-8 -*-
__author__ = 'oukohou'
__time__ = '17-6-12 上午9:46'

"""
If this runs wrong, don't ask me, I don't know why;
If this runs right, thank god, and I don't know why.
Maybe the answer, my friend, is blowing in the wind.
"""

import tensorflow as tf
import os
import time
from train.CNN_for_tfrecords import Flags, CNN_model
from tensorflow.python.platform import gfile

height = Flags.height
width = Flags.width
IMAGE_PIXELS = height * width * 3  # default


def read_decode_tfrecords(records_path, num_epochs=1020, batch_size=Flags.batch_size, num_threads=2):
    if gfile.IsDirectory(records_path):
        records_path = [os.path.join(records_path, i) for i in os.listdir(records_path)]
    else:
        records_path = [records_path]
    records_path_queue = tf.train.string_input_producer(records_path, seed=123,
                                                        # num_epochs=num_epochs,
                                                        name="string_input_producer")
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(records_path_queue, name="serialized_example")
    features = tf.parse_single_example(serialized=serialized_example,
                                       features={"img_raw": tf.FixedLenFeature([], tf.string),
                                                 "label": tf.FixedLenFeature([], tf.int64),
                                                 "height": tf.FixedLenFeature([], tf.int64),
                                                 "width": tf.FixedLenFeature([], tf.int64),
                                                 "depth": tf.FixedLenFeature([], tf.int64)},
                                       name="parse_single_example")
    image = tf.decode_raw(features["img_raw"], tf.uint8, name="decode_raw")
    image.set_shape([IMAGE_PIXELS])
    image = tf.cast(image, tf.float32) * (1.0 / 255) - 0.5
    label = tf.cast(features["label"], tf.int32)
    # images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, num_threads=num_threads,
    #                                         name="shuffle_bath", capacity=1020, min_after_dequeue=64)
    return image, label


def test_model(images_, labels_):
    # one_hot_test_labels = tf.one_hot(indices=tf.cast(labels_, tf.int32), depth=Flags.class_num,
    #                                  on_value=1, off_value=0, name="ont_hot_label_test")
    prediction = CNN_model(Flags.class_num, images_)
    # print (prediction)
    # print (one_hot_test_labels)
    # loss = tf.losses.softmax_cross_entropy(one_hot_test_labels, prediction)
    correct_predictions = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    return prediction,  correct_predictions, accuracy # loss,


def run_test_model():
    with tf.Session() as sess:
        checkpoint = tf.train.get_checkpoint_state("/home/oukohou/study/tensorflow/oukohou/gong_an_pictures/train_log")
        input_checkpoint = checkpoint.model_checkpoint_path
        input_checkpoint_path = checkpoint.model_checkpoint_path + ".meta"
        print (input_checkpoint)
        # saver = tf.train.import_meta_graph(input_checkpoint_path)

        test_record_path = "/home/oukohou/study/tensorflow/oukohou/gong_an_pictures/" \
                           "data/tf_records/500_per_gender2017-06-12_10-15-03.tfrecords"
        images, labels = read_decode_tfrecords(test_record_path, num_epochs=1, batch_size=10, num_threads=1)
        test_op = test_model(images_=images, labels_=labels)

        saver = tf.train.Saver()

        # saver = tf.train.Saver(tf.global_variables())
        init_op = tf.group(tf.global_variables_initializer())#, tf.local_variables_initializer())
        sess.run(init_op)
        saver.restore(sess, input_checkpoint)

        # sess = restore_ckpt(saver, sess, Flags.logdir)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                predictions, correct_preds, accuarcies = sess.run(test_op)
                print ("test accuarcy: ", accuarcies)
        except tf.errors.OutOfRangeError as e:
            print ("Finished testing for %d images." % (200))
        finally:
            coord.request_stop()
        coord.join(threads=threads)
        sess.close()


def main(_):
    run_test_model()


if __name__ == "__main__":
    tf.app.run(main)
