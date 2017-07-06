# -*- coding: utf-8 -*-
__author__ = 'oukohou'
__time__ = '17-6-6 上午9:16'

"""
If this runs wrong, don't ask me, I don't know why;
If this runs right, thank god, and I don't know why.
Maybe the answer, my friend, is blowing in the wind.
"""

import tensorflow as tf
import os
import time
from tensorflow.python.platform import gfile

tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_integer("num_epochs",
                            default_value=1020,
                            docstring="An integer (optional)."
                                      " If specified, `string_input_producer' produces each string "
                                      "from `string_tensor` `num_epochs` times before generating an"
                                      " `OutOfRange` error.")
tf.app.flags.DEFINE_string("record_path",
                           default_value="../data/tf_records/1000_per_gender2017-06-06_09-50-05.tfrecords",
                           docstring="path of tfrecords. ")
tf.app.flags.DEFINE_integer("batch_size", 40, "batch size.")
tf.app.flags.DEFINE_integer("class_num", 2, "classes of images, default to 5")
tf.app.flags.DEFINE_integer("learning_rate", 0.01, "learning rate.")
tf.app.flags.DEFINE_string("log_dir", "../train_log", "directory to save checkpoints.")
tf.app.flags.DEFINE_integer("total_epochs", 1020, "how many epochs in total to run train_op.")
tf.app.flags.DEFINE_integer("height", 188, "height")
tf.app.flags.DEFINE_integer("width", 140, "width")
# tf.app.flags.DEFINE_integer("width", 140, "width")
Flags = tf.app.flags.FLAGS

height = Flags.height
width = Flags.width
IMAGE_PIXELS = height * width * 3  # default


def read_decode_tfrecords(records_path, num_epochs=1020, batch_size=Flags.batch_size, num_threads=2):
    if gfile.IsDirectory(records_path):
        records_path = [os.path.join(records_path, i) for i in os.listdir(records_path)]
    else:
        records_path = [records_path]
    records_path_queue = tf.train.string_input_producer(records_path, seed=123,
                                                        num_epochs=num_epochs,
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
    images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, num_threads=num_threads,
                                            name="shuffle_bath", capacity=1020, min_after_dequeue=64)
    return images, labels


def CNN_model(class_num, input_image):
    input_layer = tf.reshape(input_image, [-1, height, width, 3], name="input_layer")

    pool0 = tf.layers.max_pooling2d(input_layer, pool_size=[2, 2], strides=[2, 2], name="pool0")
    conv1 = tf.layers.conv2d(pool0, filters=32, kernel_size=[3, 3], strides=(1, 1),
                             padding="valid", activation=tf.nn.relu, name="conv1")
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=[2, 2], name="pool1")
    conv2 = tf.layers.conv2d(pool1, filters=64, kernel_size=[3, 3],
                             padding="valid", activation=tf.nn.relu, name="conv2")
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=[2, 2], name="pool2")
    conv3 = tf.layers.conv2d(pool2, filters=48, kernel_size=[3, 3], padding="valid",
                             activation=tf.nn.relu, name="conv3")
    pool3 = tf.layers.max_pooling2d(conv3, pool_size=[2, 2], strides=[2, 2], name="pool3")
    pool3_flatten = tf.reshape(pool3, [-1, 7 * 10 * 48], name="pool3_flatten")
    dense = tf.layers.dense(pool3_flatten, units=1024, activation=tf.nn.relu, name="dense")
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, seed=1020, training=True, name="dropout")
    logits = tf.layers.dense(dropout, units=class_num, activation=tf.nn.relu, name="logits")  # [batch_size, class_num]
    return logits


def train_model(labels, logits, learning_rate=0.001):
    onthot_label = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=Flags.class_num,
                              on_value=1, off_value=0, name="onthot_label")
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onthot_label, logits=logits)
    global_step = tf.Variable(0, trainable=False, name="glabel_step")
    train_op = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate, name="GradientDescent").minimize(
        loss=loss, global_step=global_step
        , name="minimize")
    return train_op, global_step, loss


def run_train_model():
    with tf.Graph().as_default():
        images, labels = read_decode_tfrecords(Flags.record_path, Flags.num_epochs, Flags.batch_size)
        logits = CNN_model(Flags.class_num, images)
        train_model_op, global_step_de, loss = train_model(labels, logits, Flags.learning_rate)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        supervisor = tf.train.Supervisor(logdir=Flags.log_dir)

        with supervisor.managed_session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            global_step = 0
            try:
                while not coord.should_stop():
                    start_time = time.time()
                    _, losses, global_step = sess.run([train_model_op, loss, global_step_de])
                    end_time = time.time()
                    if global_step % 100 == 0:
                        print(
                            "for step:%d, loss: %.3f, cost time: %.3f" % (global_step, losses, end_time - start_time))
            except tf.errors.OutOfRangeError as e:
                print("Finished training for %d epochs %d steps" % (Flags.num_epochs, global_step))
            finally:
                coord.request_stop()
            coord.join(threads=threads)
            sess.close()


def main(_):
    run_train_model()


if __name__ == "__main__":
    tf.app.run(main)
