import tensorflow as tf
import numpy as np

from utils import Registry


'''
abstract/common definitions for distributions
'''
reg = Registry()
make_set = lambda lam_: sorted(set(k_ for v_ in reg.values() for k_ in lam_(v_.reg).keys()))
all_dists = lambda: make_set(lambda v_:v_.reg_dist)
all_funcs = lambda: make_set(lambda v_:v_.reg_func)
def is_valid_model(model, dist, func):
    return (dist in reg.get(model).reg.reg_dist.keys()) and\
           (func in reg.get(model).reg.reg_func.keys())

class ModelReg:
    def __init__(self):
        self.reg_dist = Registry()
        self.reg_func = Registry()
        self.arg_defs = []
        self.arg_dict = {}

    def add_arg(self, arg):
        self.arg_defs.append(arg)

def store_args(_a):
    arg_dict = vars(_a)
    for mod in reg.values():
        for arg_def in mod.reg.arg_defs:
            name = arg_def[0]
            mod.reg.arg_dict[name] = arg_dict[name]

def bind_args(parser):
    for mod in reg.values():
        for arg_def in mod.reg.arg_defs:
            name, kwargs = arg_def
            parser.add_argument('--%s'%name, **kwargs)


# for typical in-memory classification datasets like mnist
class DistClassification:
    def __init__(self, xy_): self.xy_ = xy_
    def size(self): return len(self.xy_[0])

    def summary(self):
        summ = np.unique(np.argmax(self.xy_[1], axis=1), return_counts=1)
        return dict(zip(*summ))

    def sample(self, size):
        if size>0:
            tot = len(self.xy_[0])
            inds = np.random.choice(tot, size=size)
            return [z_[inds] for z_ in self.xy_]
        else:
            return self.xy_

def test_dist():
    print(reg.keys())
    print(reg.get('QPQQ'))


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
        self.pl_y = plhd(cr_pl(dim_out))
        self.pl_w, logits_ = func(self.pl_x)
        self.loss = tf.reduce_mean(self.compute_loss(self.pl_y, logits_))
        self.w_len = self.pl_w.get_shape().as_list()[0]
        self.grad = tf.gradients(self.loss, self.pl_w)[0]
        self.sess = tf.compat.v1.Session()

    def get_size(self):
        return self.w_len

    def eval(self, w_, xy_):
        x_, y_ = xy_
        dd = {self.pl_w:w_, self.pl_x:x_, self.pl_y:y_}
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


import model_cifar10, model_mnist, model_toy
reg.put('cifar10', model_cifar10)
reg.put('mnist', model_mnist)
reg.put('toy', model_toy)


if __name__ == '__main__':
    test_dist()
    test_func()
