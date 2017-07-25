# -*- coding: utf-8 -*-
__author__ = 'oukohou'
__time__ = '17-6-12 下午3:56'

"""
If this runs wrong, don't ask me, I don't know why;
If this runs right, thank god, and I don't know why.
Maybe the answer, my friend, is blowing in the wind.
"""

import os

import tensorflow as tf

from tensorflow.python.platform import gfile

from train.CNN_for_tfrecords import Flags

test_record_path = "../data/tf_records/500_per_gender2017-06-12_10-15-03.tfrecords"
test_batch_size = 500
total_test_images = 6000
height = Flags.height
width = Flags.width
IMAGE_PIXELS = height * width * 3  # default


def read_decode_tfrecords(records_path, num_epochs=1, batch_size=Flags.batch_size, num_threads=1):
    if gfile.IsDirectory(records_path):
        records_path = [os.path.join(records_path, i) for i in os.listdir(records_path)]
    else:
        records_path = [records_path]
    records_path_queue = tf.train.string_input_producer(records_path, seed=123,
                                                        num_epochs=None,
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
                                            name="shuffle_bath", capacity=1020, min_after_dequeue=50)
    return images, labels


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
    conv3 = tf.layers.conv2d(pool2, filters=128, kernel_size=[3, 3], padding="valid",
                             activation=tf.nn.relu, name="conv3")
    pool3 = tf.layers.max_pooling2d(conv3, pool_size=[2, 2], strides=[2, 2], name="pool3")
    pool3_flatten = tf.reshape(pool3, [-1, 7 * 10 * 48], name="pool3_flatten")
    dense = tf.layers.dense(pool3_flatten, units=1024, activation=tf.nn.relu, name="dense")
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, seed=1020, training=True, name="dropout")
    logits = tf.layers.dense(dropout, units=class_num, activation=tf.nn.relu, name="logits")  # [batch_size, class_num]
    return logits


def test_model(labels, logits):
    onthot_label = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=Flags.class_num,
                              on_value=1, off_value=0, name="onthot_label")
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onthot_label, logits=logits)
    global_step = tf.Variable(0, trainable=False, name="glabel_step")

    print("labels:", labels)
    print("logits:", logits)

    correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(onthot_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return global_step, loss, correct_predictions, accuracy


def run_test_model():
    checkpoint = tf.train.get_checkpoint_state("../train_log")
    input_checkpoint = checkpoint.model_checkpoint_path
    input_checkpoint_path = input_checkpoint + ".meta"
    print(input_checkpoint)
    # saver = tf.train.import_meta_graph(input_checkpoint_path)


    images, labels = read_decode_tfrecords(test_record_path, num_epochs=1, batch_size=test_batch_size, num_threads=1)

    logits = CNN_model(Flags.class_num, images)
    global_step, loss, correct_predictions, accuracy = test_model(labels=labels, logits=logits)

    saver = tf.train.Saver()

    # saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # init_op2 = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(init_op)
        # sess.run(init_op2)

        saver.restore(sess, input_checkpoint)
        step = 0
        accuracy_sum = 0.0
        try:
            while not coord.should_stop() and step < total_test_images / test_batch_size:
                predictions = sess.run(logits)
                # print ("predictions: ", predictions)

                correct_preds = sess.run(correct_predictions)
                # print ("correct_preds: ", correct_preds)

                accuarcies = sess.run(accuracy)
                print("test accuarcy for batch %d : %f " % (step, accuarcies))
                accuracy_sum += accuarcies
                step += 1
        except tf.errors.OutOfRangeError as e:
            print("Finished testing for %d images." % total_test_images)
        finally:
            coord.request_stop()
            print("average accuracy of test is : %f ." % sess.run(tf.div(accuracy_sum, step)))
        coord.join(threads=threads)
        sess.close()


def main(_):
    run_test_model()


if __name__ == "__main__":
    tf.app.run(main)
