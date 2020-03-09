import tensorflow as tf

import utilities.data as du
from . import model


def get_data():
    (x_train, y_train), _ = du.get_dataset('cifar10')
    return x_train, y_train

get_data.num_classes = 10
model.datasets.put('cifar10', get_data)


'''
definitions for functions
'''
Eval = model.EvalClassification
params = model.params


def conv_net(x, keep_prob, f1,f2,f3,f4, w1,b1, w2,b2, w3,b3, w4,b4):
    # 1, 2
    conv1 = tf.nn.conv2d(x, f1, strides=[1,1,1,1], padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # conv1 = tf.layers.batch_normalization(conv1)

    # 3, 4
    conv2 = tf.nn.conv2d(conv1, f2, strides=[1,1,1,1], padding='SAME')
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # conv2 = tf.layers.batch_normalization(conv2)

    # 5, 6
    conv3 = tf.nn.conv2d(conv2, f3, strides=[1,1,1,1], padding='SAME')
    conv3 = tf.nn.relu(conv3)
    conv3 = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # conv3 = tf.layers.batch_normalization(conv3)

    # 7, 8
    conv4 = tf.nn.conv2d(conv3, f4, strides=[1,1,1,1], padding='SAME')
    conv4 = tf.nn.relu(conv4)
    conv4 = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # conv4 = tf.layers.batch_normalization(conv4)

    # 9
    flat = tf.contrib.layers.flatten(conv4)
    fully_connected = lambda xx,actv,ww,bb: actv(xx@ww+bb)
    # 10
    full1 = fully_connected(flat, tf.nn.relu, w1,b1)
    full1 = tf.nn.dropout(full1, keep_prob)
    # full1 = tf.layers.batch_normalization(full1)

    # 11
    full2 = fully_connected(full1, tf.nn.relu, w2,b2)
    full2 = tf.nn.dropout(full2, keep_prob)
    # full2 = tf.layers.batch_normalization(full2)

    # 12
    full3 = fully_connected(full2, tf.nn.relu, w3,b3)
    full3 = tf.nn.dropout(full3, keep_prob)
    # full3 = tf.layers.batch_normalization(full3)

    # # 13
    # full4 = tf.contrib.layers.fully_connected(inputs=full3, num_outputs=1024, activation_fn=tf.nn.relu)
    # full4 = tf.nn.dropout(full4, keep_prob)
    # full4 = tf.layers.batch_normalization(full4)

    return fully_connected(full3, tf.identity, w4,b4)


def reg_func(func):
    lam = lambda: Eval(func, [32,32,3], 10)
    model.funcs.put(func.__name__, lam)
    return func

@reg_func
def linear(x_):
    w_, w, b = params([32*32*3,10], 10)
    return w_, tf.contrib.layers.flatten(x_)@w+b

@reg_func
def relu(x_):
    w_, w1, b1, w2, b2 = params([32*32*3,512],512, [512,10],10)
    flat = tf.contrib.layers.flatten(x_)
    return w_, tf.nn.relu(flat@w1+b1)@w2+b2

@reg_func
def conv(x_):
    w_, *weights = params([3,3,3,64], [3,3,64,128],
                          [5,5,128,256], [5,5,256,512],
                          [2048,128], 128, [128,256], 256,
                          [256,512], 512, [512,10], 10)
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
                                                  # train_dict    # test_dict
    return w_, conv_net(x_, keep_prob, *weights), {keep_prob:.7}, {keep_prob:1.}
