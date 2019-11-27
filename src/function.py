import tensorflow as tf
from utils import Registry


reg = Registry()
def register_(func):
    inner = lambda: Evaluator(func)
    inner.__name__ = func.__name__
    reg.reg(inner)


pl = lambda sh_: tf.compat.v1.placeholder(tf.float32, shape=sh_)
sm = tf.nn.softmax_cross_entropy_with_logits_v2

# create a single parameter vector and split it to a bunch of vars
def params(*shapes):
    size_ = lambda shape: shape if isinstance(shape, int) else shape[0]*shape[1]
    w_ = pl(sum([size_(shape) for shape in shapes]))
    ret, start = [w_], 0
    for shape in shapes:
        vv = w_[start : start+size_(shape)]
        if not isinstance(shape, int): vv = tf.compat.v1.reshape(vv,shape)
        ret.append(vv)
    return ret


class Evaluator:
    def __init__(self, func):
        self.pl_x = pl((None, 784))
        self.pl_y = pl((None, 10))
        self.pl_w, logits_ = func(self.pl_x)
        self.loss = tf.reduce_mean(sm(self.pl_y, logits_))
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


@register_
def linear0(x_):
    w_, w, b = params((784,10), 10)
    return w_, x_@w+b

@register_
def linear1(x_):
    w_, w1, b1, w2, b2 = params((784,500), 500, (500,10), 10)
    return w_, (x_@w1+b1)@w2+b2

@register_
def relu1(x_):
    w_, w1, b1, w2, b2 = params((784,500), 500, (500,10), 10)
    return w_, tf.nn.relu(x_@w1+b1)@w2+b2
