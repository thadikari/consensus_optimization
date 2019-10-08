import matplotlib.pyplot as plt
import numpy as np


fwx = lambda w_, x_: (w_/np.sqrt(x_)-np.sqrt(x_))**2
gwf = lambda w_, x_: 2*(w_/x_-1)

FF = lambda w_: 0.5*(fwx(w_,1) + fwx(w_,4))

ww = np.arange(-2,8,0.1)

def get_grads(w_):
    count = 9 if np.random.rand() > .5 else 1
    return grad_gen(w_, 1, count), grad_gen(w_, 4, 10-count)

def grad_gen(w_, loc, count):
    x_ = np.random.normal(loc=loc, scale=0.3, size=count)
    return sum(gwf(w_, x_))/count, count

def grad_combine_equal(grad_confs):
    grads, confs = zip(*grad_confs)
    return sum(grads)/len(grads)

def grad_combine_conf(grad_confs):
    grads, confs = zip(*grad_confs)
    grads, confs = np.array(grads), np.array(confs)
    confs = confs/sum(confs)
    return grads@confs

def run_sample(grad_combine):
    w_curr = 10.6
    step = 0.1
    ll = [w_curr]
    for _ in range(50):
        w_curr -= step*grad_combine(get_grads(w_curr))
        ll.append(w_curr)
    return np.array(ll)


def main():
    ax1, ax2 = plt.subplot(121), plt.subplot(122)
    def prod(grad_combine, label):
        w_ = run_sample(grad_combine)
        ax1.plot(w_, label=label)
        ax2.plot(FF(w_), label=label)

    prod(grad_combine_equal, 'equal')
    prod(grad_combine_conf, 'prop')

    # ax_.set_xlim(min(ww), max(ww))
    # ax_.set_ylim(0,10)
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    plt.show()


if __name__ == '__main__': main()
