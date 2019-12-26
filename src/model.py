import tensorflow as tf
import numpy as np

from utils import Registry


'''
abstract/common definitions for distributions
'''
reg_dist = Registry()


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
reg_func = Registry()


plhd = lambda sh_: tf.placeholder(tf.float32, shape=sh_)
smax = tf.compat.v1.losses.softmax_cross_entropy

# create a single parameter vector and split it to a bunch of vars
def params(*shapes):
    size_ = lambda shape: shape if isinstance(shape, int) else shape[0]*shape[1]
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
class EvalClassification:
    def __init__(self, func, dim_inp, dim_out):
        self.pl_x = plhd((None, dim_inp))
        self.pl_y = plhd((None, dim_out))
        self.pl_w, logits_ = func(self.pl_x)
        self.loss = tf.reduce_mean(smax(self.pl_y, logits_, reduction='none'))
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



import model_mnist, model_mnist

if __name__ == '__main__':
    test_dist()
    test_func()