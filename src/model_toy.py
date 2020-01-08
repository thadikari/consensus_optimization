import numpy as np
import model


reg = model.ModelReg()
reg.add_arg(('toy_sigma2', {'help':'variance within distributions', 'type':float}))

'''
definitions for distributions
'''
reg_dist = reg.reg_dist.reg


class Dist:
    def __init__(self, mu, sigma2, label, n_class, ret1h):
        self.mu = mu
        self.cov = np.eye(len(mu))*sigma2
        self.label = label
        self.n_class = n_class
        self.ret1h = ret1h

    def sample(self, size):
        if size<=0: size = 10000
        x_ = np.random.multivariate_normal(self.mu, self.cov, size)
        if self.ret1h:
            y_ = np.zeros((size,self.n_class), dtype=int)
            y_[:,self.label] = 1
        else:
            y_ = np.ones((size,1), dtype=int)*self.label
        return (x_, y_)


class QGlobal:
    def __init__(self, locals):
        self.locals = locals

    def sample(self, size):
        assert(size<0)
        xys = [local.sample(size) for local in self.locals]
        xl, yl = zip(*xys)
        return (np.vstack(xl), np.vstack(yl))


def plot_distrb(locals, Q_global):
    import matplotlib.pyplot as plt

    def plot(dst,sz):
        x1,x2 = dst.sample(sz)[0].T
        plt.scatter(x1,x2, marker='.')

    plot(Q_global, -1)
    for loc in locals: plot(loc, 500)

    plt.gca().set_aspect('equal', 'box')
    plt.grid()
    plt.show()


def require(*arg_names):
    def inner(func):
        def wrapper(*args, **kwargs):
            for name in arg_names:
                if reg.arg_dict.get(name, None) is None:
                    raise ValueError('Missing required argument: %s'%name)
            return func(*args, **kwargs)
        return wrapper
    return inner


@require('toy_sigma2')
def distinct_n(mus, ret1h=True):
    locals = [Dist(mus[i], reg.arg_dict['toy_sigma2'], i, len(mus), ret1h) for i in range(len(mus))]
    return locals, QGlobal(locals)

@reg_dist
def distinct_2_2(): return distinct_n([[1,1], [-1,-1]], False)

@reg_dist
def distinct_4_3(): return distinct_n([[1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1]])

@reg_dist
def distinct_4_2(): return distinct_n([[1,0], [0,1], [-1,0], [0,-1]])


def test_distrb():
    reg.arg_dict['toy_sigma2'] = 0.1
    dists = distinct_2_2()
    plot_distrb(*dists)
    locals, Q_global = dists
    print(Q_global.sample(-1))
    print(locals[0].sample(5))


'''
definitions for functions
'''
from model import EvalBinaryClassification as Eval
from model import params


def reg_func(dim_inp, dim_out):
    def inner(func):
        lam = lambda: Eval(func, dim_inp, dim_out)
        reg.reg_func.put(func.__name__, lam)
        return func
    return inner

@reg_func(2,1)
def linear2(x_):
    w_, w = params((2,1))
    return w_, x_@w

@reg_func(3,4)
def linear3(x_):
    w_, w, b = params((3,4), 4)
    return w_, x_@w+b

@reg_func(2,4)
def linear4(x_):
    w_, w, b = params((2,4), 4)
    return w_, x_@w+b


if __name__ == '__main__': test_distrb()
