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
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, abort, make_response, request


NUM_INPUTS = 64
NUM_STEPS = 64
NUM_HIDDEN = 128
NUM_LABELS = 2
IMAGE_SIZE = 64
INPUT_SHAPE = (1, NUM_STEPS, NUM_INPUTS)

flags = tf.app.flags
flags.DEFINE_string('model_save_path', 'rnn_ckpt/earthquake.ckpt',
                    'Path to save and restore the model.')
FLAGS = flags.FLAGS
SAVE_PATH = FLAGS.model_save_path


W = tf.Variable(tf.random_normal([2 * NUM_HIDDEN, NUM_LABELS]))
b = tf.Variable(tf.random_normal([NUM_LABELS]))
predict_data = tf.placeholder(tf.float32, shape=INPUT_SHAPE)


def predict_single_img():
    return [np.array(np.random.randint(0, 100, 4096) / 100)
            for _ in range(1000)]


def BiRNN(x, weights, biases):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, NUM_INPUTS])
    x = tf.split(x, NUM_STEPS)
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_HIDDEN, forget_bias=1.0)
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_HIDDEN, forget_bias=1.0)
    outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights) + biases


prediction = BiRNN(predict_data, W, b)
saver = tf.train.Saver()
saver_folder = os.path.dirname(SAVE_PATH)
model_file = tf.train.latest_checkpoint(saver_folder)
print("Get model file from path: ", model_file)
sess = tf.Session(graph=tf.get_default_graph())
saver.restore(sess, model_file)
print("Model restore successfully, begin to predicting..............")

app = Flask(__name__)


def predict_waveform(data):
    expected_len = INPUT_SHAPE[1] * INPUT_SHAPE[2]
    real_len = len(data)
    if real_len != expected_len:
        return "输入数据格式错误， 长度应为： " + str(expected_len) + " 实际为： " + str(real_len)
    data = data.reshape(INPUT_SHAPE)
    res = sess.run(prediction, feed_dict={predict_data: data})
    result = np.argmax(res, 1)[0]
    return result


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.route('/input', methods=['POST'])
def input_and_predict_waveform():
    if not request.json or not 'waveform' in request.json:
        abort(400)
    waveform = np.array([float(x.strip()) for x in request.json['waveform'].split(',')])
    return jsonify({'predict_result': predict_waveform(waveform)}), 201


@app.route('/predict')
def predict():
    result = []
    predict_imgs = predict_single_img()
    for img in predict_imgs:
        result.append(predict_waveform(img))
    return "检测结果为： " + str(result) + " => " + str(sum(result))


@app.route('/')
def index():
    return '====== 欢迎使用深度学习地震波形P波检测程序！======='

# sess.close()


if __name__ == '__main__':
    app.run(host='0.0.0.0')
