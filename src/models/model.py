import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utilities'))

import tensorflow as tf
import numpy as np

import utilities as ut

datasets = ut.Registry()
funcs = ut.Registry()
arg_defs = []
arg_dict = {}

def add_arg(arg):
    arg_defs.append(arg)

def store_args(_a):
    dd = vars(_a)
    for arg_def in arg_defs:
        name = arg_def[0]
        arg_dict[name] = dd[name]

def bind_args(parser):
    for arg_def in arg_defs:
        name, kwargs = arg_def
        parser.add_argument('--%s'%name, **kwargs)


'''
abstract/common definitions for functions
'''


VAR_SCOPE = 'varcollectorjustadummyname'

def var_collector(func):
    def wrapper(*args, **kwargs):
        with tf.compat.v1.variable_scope(VAR_SCOPE):
            rets = func(*args, **kwargs)
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=VAR_SCOPE)
        return var_list, rets
    return wrapper


plhd = lambda sh_: tf.placeholder(tf.float32, shape=sh_)
smax = tf.compat.v1.losses.softmax_cross_entropy

# assignment to a list of tf.variables from a single placeholder
def create_assign_op(var_list):
    size_ = lambda tnsr: np.prod(tnsr.get_shape().as_list())
    pl_len = sum([size_(var_) for var_ in var_list])
    plh = plhd(pl_len)
    ops, start = [], 0
    for var_ in var_list:
        end = start+size_(var_)
        vec = plh[start : end]
        pl_var = tf.compat.v1.reshape(vec, var_.get_shape())
        ops.append(var_.assign(pl_var))
        start = end
    return tf.group(ops), plh, pl_len


'''
Some of the grads could be None if batch_normalization is used.
This is b/c moving_avg, moving_variance are not trainable varaibles,
but only are two parameters maintained for the testing purposes.
E.g.: If only one sample is used in testing variance for that would be zero,
so can't cal variance when testing, need to keep track of that while training.
Also, need to keep track of the varince for each worker.

https://stackoverflow.com/questions/55310934/why-moving-mean-and-moving-variance-not-in-tf-trainable-variables
https://stackoverflow.com/a/45420579/1551308: These 2048 parameters are in fact [gamma weights, beta weights, moving_mean(non-trainable), moving_variance(non-trainable)]
'''
def create_grad_vec(grads, var_list):
    vecs = []
    for grad,var_ in zip(grads, var_list):
        tnsr = tf.zeros_like(var_) if grad is None else grad
        vecs.append(tf.reshape(tnsr, [-1]))
    return tf.concat(vecs, 0)


# for typical regression problems

# for typical classification problems like mnist
class Evaluator:
    def __init__(self, func, dim_inp, dim_out):
        cr_pl = lambda dim_: ([None]+dim_) if isinstance(dim_, list) else (None, dim_)
        self.pl_x = plhd(cr_pl(dim_inp))
        self.pl_y = plhd((None))
        y1h = tf.one_hot(tf.cast(self.pl_y, tf.int32), dim_out)
        var_list, rets = func(self.pl_x)
        logits_, self.train_args, self.test_args = rets if isinstance(rets, tuple) else (rets, {}, {})
        self.loss = tf.reduce_mean(self.compute_loss(y1h, logits_))

        self.grad = create_grad_vec(tf.gradients(self.loss, var_list), var_list)
        self.assign_op, self.pl_w, self.pl_len = create_assign_op(var_list)

        self.sess = tf.compat.v1.Session()

    def get_size(self):
        return self.pl_len

    def eval(self, w_, xy_, testing=False):
        x_, y_ = xy_
        self.sess.run(self.assign_op, feed_dict={self.pl_w:w_})
        dd = {self.pl_x:x_, self.pl_y:y_,\
              **(self.test_args if testing else self.train_args)}
        loss, grad = self.sess.run([self.loss, self.grad], feed_dict=dd)
        return loss, grad

class EvalClassification(Evaluator):
    def compute_loss(self, label, logits):
        return smax(label, logits, reduction='none')

class EvalBinaryClassification(Evaluator):
    def compute_loss(self, label, logits):
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits)


def test_func():
    '''
    Test for @params function. Should print the following.
    [0. 1. 2. 3. 4. 5. 6. 7. 8.]
    [0.]
    [1. 2.]
    [[3. 4.]
     [5. 6.]
     [7. 8.]]
    '''
    w_, w1, w2, w3 = params(1, 2, (3,2))
    with tf.train.MonitoredTrainingSession() as ss:
        lenw = w_.get_shape().as_list()[0]
        output = ss.run([w_, w1, w2, w3], feed_dict={w_:range(lenw)})
        print(*output, sep = '\n')


if __name__ == '__main__':
    test_dist()
    test_func()
