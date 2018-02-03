# -*- coding: utf-8 -*-

"""
地震波形P波起点检测

   使用Bi-LSTM(双向长短时记忆循环神经网络)网络模型对40s内的Z波进行自动检测，识别其中有P波起点的波形，标记为‘1’。

用法：
    1. predict folder:
    python waveform_classification_rnn.py \
       --model_path=rnn_ckpt/rnn_earthquake.ckpt
       --predict_mode=folder\
       --predict_folder=earthquake/w/20080801\
       --result_folder=result

    2. predict single file:
    python waveform_classification_rnn.py \
       --model_path=rnn_ckpt/rnn_earthquake.ckpt \
       --predict_mode=file \
       --predict_file=earthquake/w/20080801/XX.JJS.2008214000000.BHZ \
       --result_folder=result

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import obspy
import numpy as np
import tensorflow as tf

NUM_INPUTS = 64
NUM_STEPS = 64
NUM_HIDDEN = 128
NUM_LABELS = 2
IMAGE_SIZE = 64
INPUT_SHAPE = (1, NUM_STEPS, NUM_INPUTS)

flags = tf.app.flags
flags.DEFINE_string('model_path', 'rnn_ckpt/rnn_earthquake.ckpt',
                    'Path to load the model.')
flags.DEFINE_string('predict_file', r'C:\codes\earthquake\data\XX.MXI.2008227000000.BHZ',
                    'Path of waveform file to predict.')
flags.DEFINE_string('predict_folder', '',
                    'Path of waveform folder to predict.')
flags.DEFINE_string('predict_mode', 'file',
                    'Predict single file or a folder: file or folder')
flags.DEFINE_string('result_folder', 'result',
                    'Path to write predict results.')

FLAGS = flags.FLAGS
MODEL_PATH = FLAGS.model_path
assert os.path.exists(os.path.dirname(MODEL_PATH)), "The path to load the model is not esxtis!"

PREDICT_MODE = FLAGS.predict_mode
assert PREDICT_MODE in ["file", "folder"], "The predict mode should be file or folder."

if PREDICT_MODE is "file":
    predict_file = FLAGS.predict_file
    assert os.path.isfile(predict_file), "Path of waveform file to predict is not exists!"
    assert not os.path.isdir(predict_file), "Path of waveform file to predict should not be folder!"
    assert predict_file.endswith('.BHZ'), "We only support BHZ file."

elif PREDICT_MODE is "folder":
    assert os.path.isdir(FLAGS.predict_folder), "Path of waveform folder to predict is not an folder!"
    assert not os.path.isfile(FLAGS.predict_folder), "Path of waveform file to predict should not be a file!"

RESULT_FOLDER = FLAGS.result_folder
if not os.path.exists(RESULT_FOLDER):
    print("Make dir: ", RESULT_FOLDER)
    os.mkdir(RESULT_FOLDER)


# 数据预处理， 去趋势采用去均值的方式，滤波器采用带通滤波.
def pre_process(src_file, fmin=0.5, fmax=20.0):
    st = obspy.read(src_file).copy()

    # 去趋势
    st.detrend('constant')

    # 带通滤波
    st.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=2.0, zerophase=True)

    # 处理后的数据
    return st


def max_min(row):
    max_val, min_val = max(row), min(row)
    diff = max_val - min_val
    if diff == 0:
        return np.zeros(len(row))
    else:
        return [(x - min_val) / diff for x in row]


def read_and_input(waveform_file):
    print("Preprocessing: " + waveform_file)
    st = pre_process(waveform_file)
    tr = st[0]
    starttime = tr.stats.starttime
    datas = np.array(tr.data)
    total_count = len(datas)
    row_count = total_count // 4000
    temp = datas[: row_count * 4000].reshape((-1, 4000))
    return [max_min(x) for x in temp], starttime


def BiRNN(x, weights, biases):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, NUM_INPUTS])
    x = tf.split(x, NUM_STEPS)
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_HIDDEN, forget_bias=1.0)
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_HIDDEN, forget_bias=1.0)
    outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights) + biases


def main(_):
    W = tf.Variable(tf.random_normal([2 * NUM_HIDDEN, NUM_LABELS]))
    b = tf.Variable(tf.random_normal([NUM_LABELS]))
    predict_data = tf.placeholder(tf.float32, shape=INPUT_SHAPE)
    prediction = BiRNN(predict_data, W, b)
    saver = tf.train.Saver()
    saver_folder = os.path.dirname(MODEL_PATH)
    model_file = tf.train.latest_checkpoint(saver_folder)
    print("Get model file from path: ", model_file)
    sess = tf.Session(graph=tf.get_default_graph())
    saver.restore(sess, model_file)
    print("Model restore successfully, begin to predicting..............")

    def predict_input(data):
        data = np.concatenate((data, np.zeros(96)))
        expected_len = INPUT_SHAPE[1] * INPUT_SHAPE[2]
        real_len = len(data)
        assert real_len == expected_len
        data = data.reshape(INPUT_SHAPE)
        raw_prediction = sess.run(tf.nn.softmax(prediction), feed_dict={predict_data: data})
        predict_result = np.argmax(raw_prediction, 1)[0]
        probability = raw_prediction[0][predict_result]
        return (predict_result, probability)

    def predict_waveform_file(waveform_file):
        result_file = os.path.join(RESULT_FOLDER, os.path.basename(waveform_file).replace("BHZ", "txt"))
        print(os.path.basename(waveform_file))
        print(result_file)
        if os.path.exists(result_file):
            os.remove(result_file)
        predict_inputs, starttime = read_and_input(waveform_file)
        print("Writing result to " + result_file)
        with open(result_file, "a") as f:
            for index, input in enumerate(predict_inputs):
                predict_result, probability = predict_input(input)
                print(index, predict_result, probability)
                if predict_result == 1:
                    current_time = starttime + index * 40
                    content = str(current_time) + '\t' + str(probability) + '\n'
                    f.write(content)

    print("Predict mode: " + PREDICT_MODE)
    if PREDICT_MODE == "file":
        waveform_file = FLAGS.predict_file
        print("Predicting waveform file: " + waveform_file)
        predict_waveform_file(waveform_file)
    elif PREDICT_MODE == "folder":
        waveform_folder = FLAGS.predict_folder
        print("Predicting waveform folder: " + waveform_folder)
        waveform_files = [os.path.join(waveform_folder, x)
                          for x in os.listdir(waveform_folder)
                          if x.endswith('.BHZ')]
        for waveform_file in waveform_files:
            predict_waveform_file(waveform_file)

    sess.close()


if __name__ == '__main__':
    tf.app.run()
