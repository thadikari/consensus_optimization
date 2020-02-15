import tensorflow as tf
import numpy as np

from . import common

datasets = common.Registry()
funcs = common.Registry()
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

plhd = lambda sh_: tf.placeholder(tf.float32, shape=sh_)
smax = tf.compat.v1.losses.softmax_cross_entropy

# create a single parameter vector and split it to a bunch of vars
def params(*shapes):
    size_ = lambda shape: shape if isinstance(shape, int) else np.prod(shape)
    w_ = plhd(sum([size_(shape) for shape in shapes]))
    ret, start = [w_], 0
    for shape in shapes:
        end = start+size_(shape)
        vv = w_[start : end]
        if not isinstance(shape, int): vv = tf.compat.v1.reshape(vv,shape)
        ret.append(vv)
        start = end
    return ret


# for typical regression problems

# for typical classification problems like mnist
class Evaluator:
    def __init__(self, func, dim_inp, dim_out):
        cr_pl = lambda dim_: ([None]+dim_) if isinstance(dim_, list) else (None, dim_)
        self.pl_x = plhd(cr_pl(dim_inp))
        self.pl_y = plhd((None))
        y1h = tf.one_hot(tf.cast(self.pl_y, tf.int32), dim_out)
        self.pl_w, logits_, *args = func(self.pl_x)
        self.train_args, self.test_args = args if len(args)>0 else ({}, {})
        self.loss = tf.reduce_mean(self.compute_loss(y1h, logits_))
        self.w_len = self.pl_w.get_shape().as_list()[0]
        self.grad = tf.gradients(self.loss, self.pl_w)[0]
        self.sess = tf.compat.v1.Session()

    def get_size(self):
        return self.w_len

    def eval(self, w_, xy_, testing=False):
        x_, y_ = xy_
        dd = {self.pl_w:w_, self.pl_x:x_, self.pl_y:y_,\
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
