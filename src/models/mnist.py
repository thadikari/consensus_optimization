import tensorflow as tf
import numpy as np

from . import data_utils as du
from . import model


reg = model.ModelReg()
model.reg.put('fashion_mnist', reg)


def process_data():
    (x_train, y_train), _ = du.get_dataset('fashion_mnist')
    x_train = x_train.reshape([-1, 784])
    x_train, y_train = du.permute(x_train, y_train)
    Q_global = Dist((x_train, y_train))
    return x_train, y_train, Q_global

'''
definitions for distributions
'''
Dist = model.DistClassification
reg_dist = reg.reg_dist.reg

@reg_dist
def identical_10():
    x_, y_, Q_global = process_data()
    locals = [Dist((x_, y_)) for _ in range(10)]
    return locals, Q_global


@reg_dist
def distinct_10():
    x_, y_, Q_global = process_data()
    indss = [y_==cls for cls in range(10)]
    #count = min(inds.sum() for inds in indss)
    locals = [Dist((x_[inds], y_[inds])) for inds in indss]
    return locals, Q_global


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
definitions for functions
'''
Eval = model.EvalClassification
params = model.params


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
