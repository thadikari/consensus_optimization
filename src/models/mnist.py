import tensorflow as tf

import utilities.data as du
import utilities.models.mnist as mnist
from . import model


def get_data(name):
    def inner():
        (x_train, y_train), _ = du.get_dataset(name)
        x_train = x_train.reshape([-1, 784])
        return x_train, y_train
    inner.num_classes = 10
    return inner

register = lambda name: model.datasets.put(name, get_data(name))
register('fashion_mnist')
register('mnist')


''' needs refactoring
# 'PQQQ', 'QPQQ', 'QQPQ', 'QQQP'
def type_1_3(position_of_P):
    x_, y_ = process_data()
    indss = [y_==cls for cls in range(4)]

    make_dist = lambda ind_: Dist((x_[ind_], y_[ind_]))

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


'''
definitions for functions
'''
Eval = model.EvalClassification
dense = tf.layers.dense


def reg_func(func):
    lam = lambda: Eval(model.var_collector(func), 784, 10)
    model.funcs.put(func.__name__, lam)
    return func

@reg_func
def linear0(x_):
    return dense(x_, 10, activation=None)

@reg_func
def linear1(x_):
    x_ = dense(x_, 500, activation=None)
    x_ = dense(x_, 10, activation=None)
    return x_

@reg_func
def relu1(x_):
    x_ = dense(x_, 500, activation=tf.nn.relu)
    x_ = dense(x_, 10, activation=None)
    return x_

@reg_func
def conv1(x_):
    return mnist.create_conv(x_)
