# -*- coding: future_fstrings -*-

import tensorflow as tf
import numpy as np
import argparse
import json

from mnist import get_data


class Evaluator:
    def __init__(self):
        self.w_len = 784*10+10
        pl = lambda sh_: tf.compat.v1.placeholder(tf.float32, shape=sh_)
        self.pl_w = pl(self.w_len)
        self.pl_x = pl((None, 784))
        self.pl_y = pl((None, 10))
        self.loss = self.func(self.pl_w, self.pl_x, self.pl_y)
        self.grad = tf.gradients(self.loss, self.pl_w)[0]
        self.sess = tf.compat.v1.Session()

    def func(self, w_, x_, y_):
        reshape = tf.compat.v1.reshape
        w, b = reshape(w_[:784*10], (784,10)), w_[784*10:]
        #print(w.get_shape(), b.get_shape())
        logits = x_@w+b
        sm = tf.nn.softmax_cross_entropy_with_logits_v2
        #return tf.reduce_mean(tf.square(logits - y_))
        return tf.reduce_mean(sm(y_, logits))

    def get_size(self):
        return self.w_len

    def eval(self, w_, x_, y_):
        dd = {self.pl_w:w_, self.pl_x:x_, self.pl_y:y_}
        loss, grad = self.sess.run([self.loss, self.grad], feed_dict=dd)
        return loss, grad


class Worker:
    def __init__(self, eval, xy_):
        self.tot = len(xy_[0])
        assert(self.tot>5400)
        self.xy_ = xy_
        self.eval = eval

    def get_loss(self, weights):
        loss, _ = self.eval.eval(weights, *self.xy_)
        return loss

    def get_num_samples(self):
        if args.method=='bern':
            num_samples = args.num_samples if np.random.rand() < args.dist_param else 1
        elif args.method=='gauss':
            num_samples = int(np.random.normal(loc=args.num_samples, scale=args.dist_param))
            num_samples = max(1, min(self.tot, num_samples))
        return num_samples

    def prep_data(self, num_samples=-1):
        self.num_samples = num_samples if num_samples>0 else self.get_num_samples()
        self.inds = np.random.choice(self.tot, size=self.num_samples)

    def get_grad(self, weights):
        x_, y_ = [z_[self.inds] for z_ in self.xy_]
        loss, grad = self.eval.eval(weights, x_, y_)
        return loss, grad, self.num_samples


def grad_combine_equal(grads, num_samples):
    return sum(grads)/len(grads)

def grad_combine_conf(grads, num_samples):
    grads, confs = np.array(grads), np.array(num_samples)
    confs = confs/sum(confs)
    print(num_samples)
    return confs@grads


class Scheme():
    def __init__(self, workers, w_curr, grad_combine):
        self.workers = workers
        self.w_curr = w_curr
        self.comb = grad_combine
        self.history = [self.loss()]

    def loss(self):
        losses = [worker.get_loss(self.w_curr) for worker in self.workers]
        return sum(losses)/len(losses)

    def step(self):
        step = 0.1
        worker_outs = [worker.get_grad(self.w_curr) for worker in self.workers]
        losses, grads, num_samples = zip(*worker_outs)
        curr_loss = sum(losses)/len(losses)
        self.w_curr -= step*self.comb(grads, num_samples)
        tot_loss = self.loss()
        self.history.append(tot_loss)
        return tot_loss


def main():
    eval = Evaluator()
    w_start = np.random.normal(size=eval.get_size())
    workers = [Worker(eval, xy_) for xy_ in get_data(args.identical)]
    sc = lambda comb: Scheme(workers, np.copy(w_start), comb)
    schemes = {'Equal':sc(grad_combine_equal), 'Weighted':sc(grad_combine_conf)}

    for t in range(args.num_iters):
        for i in range(len(workers)):
            if args.method=='round':
                numsam = args.num_samples if t%len(workers)==i else 1
            else:
                numsam = -1
            workers[i].prep_data(numsam)

        print([schemes[scheme].step() for scheme in schemes])

    run_id = f'run_{args.method}_{args.num_samples}_{args.identical}'
    with open('%s.json'%run_id, 'w') as fp_:
        dd = {scheme:schemes[scheme].history for scheme in schemes}
        json.dump({**vars(args), **dd}, fp_, indent=4)
    # plot([scheme.history for scheme in schemes], ['Equal', 'Weighted'])


def plot(data, names):
    import matplotlib.pyplot as plt
    ax = plt.gca()
    for scheme in schemes:
        ax.plot(schemes[scheme], label=scheme)

    # ax_.set_xlim(min(ww), max(ww))
    # ax_.set_ylim(0,10)
    ax.legend(loc='best')
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('method', choices=['round', 'gauss', 'bern'])
    parser.add_argument('--num_iters', help='total iterations count', type=int, default=1000)
    parser.add_argument('--num_samples', help='num_samples in each sampling method', type=int, default=50)
    parser.add_argument('--identical', help='identical sampling across workers', action='store_true')
    parser.add_argument('--dist_param', help='sigma or true prob in gauss/bern', type=float, default=1.)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print('[Arguments]', vars(args))
    main()
