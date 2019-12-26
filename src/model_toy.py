import numpy as np
import model


reg = model.ModelReg()


'''
definitions for distributions
'''
reg_dist = reg.reg_dist.reg


class Dist:
    def __init__(self, mu, sigma2, label):
        self.mu = mu
        self.cov = [[sigma2, 0], [0, sigma2]]
        self.label = label

    def sample(self, size):
        if size<=0: size = 10000
        x_ = np.random.multivariate_normal(self.mu, self.cov, size)
        y_ = np.zeros((size,4), dtype=int)
        y_[:,self.label] = 1
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


@reg_dist
def distinct_4():
    sigma2 = .01
    c_ = lambda mu,lab: Dist(mu, sigma2, lab)
    locals = [c_([1,0], 0), c_([0,1], 1), c_([-1,0], 2), c_([0,-1], 3)]
    Q_global = QGlobal(locals)
    return locals, Q_global


def test_distrb():
    locals, Q_global = distinct_4()
    plot_distrb(locals, Q_global)
    print(Q_global.sample(-1))
    print(locals[0].sample(5))



'''
definitions for functions
'''
from model import EvalClassification as Eval
from model import params


def reg_func(func):
    lam = lambda: Eval(func, 2, 4)
    reg.reg_func.put(func.__name__, lam)
    return func

@reg_func
def linear0(x_):
    w_, w, b = params((2,4), 4)
    return w_, x_@w+b


if __name__ == '__main__': test_distrb()
