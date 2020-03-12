import tensorflow as tf

import utilities.data as du
import utilities.models.cifar as cifar
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
dense = tf.layers.dense


def reg_func(func):
    lam = lambda: Eval(model.var_collector(func), [32,32,3], 10)
    model.funcs.put(func.__name__, lam)
    return func

@reg_func
def linear(x_):
    x_ = tf.contrib.layers.flatten(x_)
    return dense(x_, 10, activation=None)

@reg_func
def relu(x_):
    x_ = tf.contrib.layers.flatten(x_)
    x_ = dense(x_, 512, activation=tf.nn.relu)
    x_ = dense(x_, 10, activation=None)
    return x_

@reg_func
def conv(x_):
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    flag_training = tf.placeholder(tf.bool)
    return cifar.create_conv10(x_, keep_prob, flag_training),\
           {keep_prob:.7, flag_training:True}, {keep_prob:1., flag_training:False}
           # train_dict,                       # test_dict
