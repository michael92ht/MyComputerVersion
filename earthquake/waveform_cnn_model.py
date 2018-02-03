"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.

python wg_digits_model.py --job=predict --predict_images_path=./predict/test_imgs --predict_type=frame

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import numpy as np
import tensorflow as tf
from sklearn import metrics


IMAGE_SIZE = 64
NUM_CHANNELS = 1
NUM_LABELS = 2
VALIDATION_SIZE = 2000     # Size of the validation set.
SEED = 66478               # Set to None for random seed.
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 500        # Number of steps between evaluations.
SAVE_FREQUENCY = 100

flags = tf.app.flags

flags.DEFINE_integer('batch_size', 32, 'batch size for model training.')
flags.DEFINE_integer('num_epochs', 10000, 'numbers of total train epochs.')
flags.DEFINE_float('base_learning_rate', 0.01, 'base learning rate at training begin.')
flags.DEFINE_string('pickled_dataset', 'C:\codes\earthquake\datasets_10\pickled', 'Path to pickled dataset.')
flags.DEFINE_boolean('balance', False, 'if the dataset is balance or not.')
flags.DEFINE_string('model_save_path', 'ckpt/earthquake.ckpt',
                    'Path to save and restore the model.')
flags.DEFINE_string('predict_images_path', 'predict',
                    'Path to the images for predict.')
flags.DEFINE_string('job', 'train',
                    'train or predict images.')


FLAGS = flags.FLAGS

assert FLAGS.job in ['train', 'predict']

NUM_EPOCHS = FLAGS.num_epochs
BATCH_SIZE = FLAGS.batch_size
SAVE_PATH = FLAGS.model_save_path
LR = FLAGS.base_learning_rate
BALANCE = FLAGS.balance


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
        predictions.shape[0])


def test_result(predictions, labels):
    test_error = error_rate(predictions, labels)
    print('Test error: %.3f%%' % test_error)
    print("Confusion Matrix: ")
    expected = np.argmax(labels, 1)
    predicted = np.argmax(predictions, 1)
    mat = metrics.confusion_matrix(expected, predicted)
    print(" \t" + "\t\t\t".join([str(x) for x in range(2)]))
    for index in range(2):
        print(str(index) + "\t" + "\t\t\t".join([str(x) for x in mat[index]]))


def get_train_data(balance=True):
    assert os.path.exists(FLAGS.pickled_dataset)
    train_images_pickled = os.path.join(FLAGS.pickled_dataset, 'train_waveform_pickled')
    train_labels_pickled = os.path.join(FLAGS.pickled_dataset, 'train_labels_pickled')
    test_images_pickled = os.path.join(FLAGS.pickled_dataset, 'test_waveform_pickled')
    test_labels_pickled = os.path.join(FLAGS.pickled_dataset, 'test_labels_pickled')

    if balance:
        train_data = np.load(train_images_pickled)
    else:
        left = np.load(train_images_pickled + "_0")
        right = np.load(train_images_pickled + "_1")
        train_data = np.concatenate((left, right))
    train_labels = np.load(train_labels_pickled)
    test_data = np.load(test_images_pickled)
    test_labels = np.load(test_labels_pickled)

    # Generate a validation set.
    validation_data = train_data[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:]

    print("Get train data done with balance is " + str(balance))
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def main(_):

    train_data_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))
    eval_data = tf.placeholder(tf.float32, shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

    predict_data = tf.placeholder(tf.float32, shape=(1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

    conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32], stddev=0.1, seed=SEED, dtype=tf.float32), name="conv1_weights")
    conv1_biases = tf.Variable(tf.zeros([32], dtype=tf.float32), name="conv1_biases")
    conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1, seed=SEED, dtype=tf.float32),
                                name="conv2_weights")
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32), name="conv2_biases")

    # begin
    conv3_weights = tf.Variable(
        tf.truncated_normal([3, 3, 64, 96], stddev=0.1, seed=SEED, dtype=tf.float32), name="conv3_weights")
    conv3_biases = tf.Variable(tf.zeros([96], dtype=tf.float32), name="conv3_biases")
    conv4_weights = tf.Variable(tf.truncated_normal([3, 3, 96, 128], stddev=0.1, seed=SEED, dtype=tf.float32), name="conv4_weights")
    conv4_biases = tf.Variable(tf.constant(0.1, shape=[128], dtype=tf.float32), name="conv4_biases")
    # end

    fc1_weights = tf.Variable(tf.truncated_normal([2048, 1024],
                                                  stddev=0.1,
                                                  seed=SEED,
                                                  dtype=tf.float32), name="fc1_weights")
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[1024], dtype=tf.float32), name="fc1_biases")
    fc2_weights = tf.Variable(tf.truncated_normal([1024, NUM_LABELS],
                                                  stddev=0.1,
                                                  seed=SEED,
                                                  dtype=tf.float32), name="fc2_weights")
    fc2_biases = tf.Variable(tf.constant(
        0.1, shape=[NUM_LABELS], dtype=tf.float32), name="fc2_biases")

    def model(data, train=False):
        conv = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv = tf.nn.conv2d(pool, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        # conv = tf.nn.conv2d(relu, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # begin
        conv = tf.nn.conv2d(pool, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv3_biases))
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv = tf.nn.conv2d(pool, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv4_biases))
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # end

        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        return tf.matmul(hidden, fc2_weights) + fc2_biases

    prediction = tf.nn.softmax(model(predict_data))

    # Predictions for the test and validation, which we'll compute less often.
    eval_prediction = tf.nn.softmax(model(eval_data))

    def eval_in_batches(data, sess):
        size = data.shape[0]
        if size < EVAL_BATCH_SIZE:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = np.ndarray(shape=(size, NUM_LABELS), dtype=np.float32)
        for begin in range(0, size, EVAL_BATCH_SIZE):
            end = begin + EVAL_BATCH_SIZE
            if end <= size:
                predictions[begin:end, :] = sess.run(
                    eval_prediction,
                    feed_dict={eval_data: data[begin:end, ...]})
            else:
                batch_predictions = sess.run(
                    eval_prediction,
                    feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        return predictions

    saver = tf.train.Saver()
    saver_folder = os.path.dirname(SAVE_PATH)

    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = get_train_data(BALANCE)
    train_size = train_labels.shape[0]
    logits = model(train_data_node, True)

    base_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_labels_node, logits=logits))

    if not BALANCE:
        # your class weights
        class_weights = tf.constant([[1.0, 10.0]])
        # deduce weights for batch samples based on their true label
        weights = tf.reduce_sum(class_weights * train_labels_node, axis=1)
        # compute your (unweighted) softmax cross entropy loss
        # apply the weights, relying on broadcasting of the multiplication
        weighted_losses = base_loss * weights
        # reduce the result to get your final loss
        base_loss = tf.reduce_mean(weighted_losses)

    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    loss = base_loss + 5e-4 * regularizers

    batch = tf.Variable(0, dtype=tf.float32)
    learning_rate = tf.train.exponential_decay(
        LR,                            # Base learning rate.
        batch * BATCH_SIZE,            # Current index into the dataset.
        train_size,                    # Decay step.
        0.95,                          # Decay rate.
        staircase=True)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=batch)

    # Predictions for the current training minibatch.
    train_prediction = tf.nn.softmax(logits)

    start_time = time.time()
    total_step = 0
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        if os.path.exists(saver_folder):
            model_file = tf.train.latest_checkpoint(saver_folder)
            print("Get model file from path: ", model_file)
            saver.restore(sess, model_file)
            print("Restore parameters from the model.")
            total_step = int(model_file.split('-')[-1])
        else:
            print('Initialized!')
        temp_step = 0
        for step in range(int(NUM_EPOCHS * train_size) // BATCH_SIZE + 1):
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            feed_dict = {train_data_node: batch_data, train_labels_node: batch_labels}

            sess.run(optimizer, feed_dict=feed_dict)

            if step % EVAL_FREQUENCY == 0:
                l, lr, predictions = sess.run([loss, learning_rate, train_prediction], feed_dict=feed_dict)
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print('Step %d (epoch %.2f), %.1f ms' %
                      (step, float(step) * BATCH_SIZE / train_size,
                       1000 * elapsed_time / EVAL_FREQUENCY))
                print('Minibatch loss: %.3f, learning rate: %.10f' % (l, lr))
                print('Minibatch error: %.3f%%' % error_rate(predictions, batch_labels))
                print('Validation error: %.3f%%' % error_rate(
                    eval_in_batches(validation_data, sess), validation_labels))
                sys.stdout.flush()

            current_epoch = int(float(step) * BATCH_SIZE / train_size)
            if current_epoch != temp_step and current_epoch % SAVE_FREQUENCY == 0:
                saver_path = saver.save(sess, save_path=SAVE_PATH, global_step=total_step + current_epoch)
                temp_step = current_epoch
                print("Save model to path: ", saver_path)

        # test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
        # print('Test error: %.3f%%' % test_error)
        test_result(eval_in_batches(test_data, sess), test_labels)


if __name__ == '__main__':
    tf.app.run()
