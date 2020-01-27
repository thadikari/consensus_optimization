from tensorflow import keras
import tensorflow as tf
import numpy as np
import os

import model


reg = model.ModelReg()


'''
download mnist data
'''
def get_mnist():
    # Keras automatically creates a cache directory in ~/.keras/datasets for
    # storing the downloaded MNIST data. This creates a race
    # condition among the workers that share the same filesystem. If the
    # directory already exists by the time this worker gets around to creating
    # it, ignore the resulting exception and continue.
    cache_dir = os.path.join(os.path.expanduser('~'), '.keras', 'datasets')
    if not os.path.exists(cache_dir):
        try:
            os.mkdir(cache_dir)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(cache_dir):
                pass
            else:
                raise

    (x_train_, y_train), (x_test_, y_test) = keras.datasets.mnist.load_data('MNIST-data')
    x_train = np.reshape(x_train_, (-1, 784)) / 255.0
    # x_test = np.reshape(x_test_, (-1, 784)) / 255.0
    assert len(x_train) == len(y_train)
    # assert len(x_test) == len(y_test)
    return x_train, y_train

def to_1hot(a):
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size),a] = 1
    return b

def permute(x_, y_, seed=None):
    p = np.random.RandomState(seed=seed).permutation(len(x_))
    return x_[p], y_[p]

def process_data():
    x_train, y_train = permute(*get_mnist())
    y_train1h = to_1hot(y_train)
    Q_global = Dist((x_train, y_train1h))
    return x_train, y_train, y_train1h, Q_global



'''
definitions for distributions
'''
from model import DistClassification as Dist
reg_dist = reg.reg_dist.reg

@reg_dist
def identical_10():
    x_, y_, y1h_, Q_global = process_data()
    locals = [Dist((x_, y1h_)) for _ in range(10)]
    return locals, Q_global


@reg_dist
def distinct_10():
    x_, y_, y1h_, Q_global = process_data()
    indss = [y_==cls for cls in range(10)]
    #count = min(inds.sum() for inds in indss)
    locals = [Dist((x_[inds], y1h_[inds])) for inds in indss]
    return locals, Q_global


# 'PQQQ', 'QPQQ', 'QQPQ', 'QQQP'
def type_1_3(position_of_P):
    x_, y_, y1h_, _ = process_data()
    indss = [y_==cls for cls in range(4)]

    make_dist = lambda ind_: Dist((x_[ind_], y1h_[ind_]))

    P_ = make_dist(indss[0])
    Q_ = make_dist(np.logical_or.reduce(indss[1:]))
    Q_global = make_dist(np.logical_or.reduce(indss))

    locals = [Q_]*4
    locals[position_of_P] = P_
    return locals, Q_global

for i in range(4):
    name = list('QQQQ')
    name[i] = 'P'
    tp = lambda: type_1_3(i)
    tp.__name__ = ''.join(name)
    reg_dist(tp)


def test_distrb():
    locals, Q_global = type_1_3(2)
    for Q_ in locals: print(Q_.size())
    for Q_ in locals: print(Q_.summary())

    locals, Q_global = distinct_10()
    assert(60000==sum(len(Q_.sample(-1)[0]) for Q_ in locals))
    for i in range(len(locals)):
        x_, y_ = locals[i].sample(-1)
        assert(len(x_)==len(y_))
        assert(np.all(np.argmax(y_, axis=1)==i))
        print(x_.shape, y_.shape)



'''
definitions for functions
'''
from model import EvalClassification as Eval
from model import params


def reg_func(func):
    lam = lambda: Eval(func, 784, 10)
    reg.reg_func.put(func.__name__, lam)
    return func

@reg_func
def linear0(x_):
    w_, w, b = params((784,10), 10)
    return w_, x_@w+b

@reg_func
def linear1(x_):
    w_, w1, b1, w2, b2 = params((784,500), 500, (500,10), 10)
    return w_, (x_@w1+b1)@w2+b2

@reg_func
def relu1(x_):
    w_, w1, b1, w2, b2 = params((784,500), 500, (500,10), 10)
    return w_, tf.nn.relu(x_@w1+b1)@w2+b2



if __name__ == '__main__': test_distrb()
