import numpy as np
import argparse
import json
import os

from graphs import make_doubly_stoch, graph_defs, eig_vals
from models import model, strategy
import utilities as ut
import utilities.file


class Worker:
    def __init__(self, eval, Q_local):
        self.eval = eval
        self.Q_local = Q_local

    def compute_loss(self, weights, samples_):
        return float(self.eval.eval(weights, samples_, testing=True)[0])

    def get_num_samples(self):
        if _a.strag_dist=='bern':
            num_samples = _a.num_samples if np.random.rand() < _a.strag_dist_param else 1
        elif _a.strag_dist=='gauss':
            num_samples = int(np.random.normal(loc=_a.num_samples, scale=_a.strag_dist_param))
            num_samples = max(1, num_samples)
        return num_samples

    def prep_straggler(self, num_samples):
        self.num_samples = num_samples if num_samples>0 else self.get_num_samples()
        self.samples = self.Q_local.sample(self.num_samples)

    def compute_grad(self, weights):
        loss, grad = self.eval.eval(weights, self.samples)
        return loss, grad, self.num_samples


def grad_combine_equal(grads, num_samples):
    ## assuming \gamma_i * len(grads) = 1
    return grads # should ideally be (len(grads)*gamma)*grads

def grad_combine_conf(grads, num_samples):
    confs = len(grads)*num_samples/sum(num_samples)
    # print(confs)
    return confs[:, np.newaxis]*grads


opts = ut.Registry()
reg_ = lambda name: (lambda cls: opts.put(name, cls))

class Optimizer:
    def __init__(self, w_init, mat_P, grad_combine):
        self.w_init = w_init
        self.numw = len(mat_P)
        self.comb = grad_combine
        self.cons = lambda arr: mat_P@arr
    def init(self): return self

@reg_('PG')
class GradientDescent_PG(Optimizer):
    def initw(self, w_):
        for wi in w_: w_[:] = self.w_init
    def apply(self, w_, g_, numsam, lrate):
        w_[:] -= lrate*self.cons(self.comb(g_, numsam))

@reg_('PWG')
class GradientDescent_PWG(Optimizer):
    def initw(self, w_):
        for wi in w_: w_[:] = self.w_init
    def apply(self, w_, g_, numsam, lrate):
        w_[:] = self.cons(w_ - lrate*self.comb(g_, numsam))

@reg_('PWG1')
class GradientDescent_PWG(Optimizer):
    def initw(self, w_): w_[:] = 0
    def apply(self, w_, g_, numsam, lrate):
        w_[:] = self.cons(w_ - lrate*self.comb(g_, numsam))

@reg_('PW')
class GradientDescent_DA(Optimizer):
    def initw(self, w_): w_[:] = 0
    def apply(self, w_, g_, numsam, lrate):
        w_[:] = self.cons(w_) - lrate*self.comb(g_, numsam)

@reg_('DA')
class DualAveraging_CL(Optimizer):
    def init(self):
        self.z_ = np.zeros([self.numw, len(self.w_init)])
        return self
    def initw(self, w_): w_[:] = 0
    def apply(self, w_, g_, numsam, lrate):
        self.z_[:] = self.cons(self.z_) + self.comb(g_, numsam)
        w_[:] = -lrate*(self.z_)


class Scheme:
    def __init__(self, workers, dim_w, Q_global, core_opt):
        self.workers = workers
        self.core_opt = core_opt
        self.Q_global = Q_global

        numw = len(workers)
        self.curr_w = np.zeros([numw, dim_w])
        self.core_opt.initw(self.curr_w)
        self.curr_g = np.zeros_like(self.curr_w)
        self.curr_numsam = np.zeros(numw)

        self.history = []
        self.var_history = []
        self.eval_global_losses()

    def eval_global_losses(self):
        samples_ = self.Q_global.sample(_a.max_loss_eval_size)
        losses = [wkr.compute_loss(wgt, samples_)
                    for wkr, wgt in zip(self.workers, self.curr_w)]
        # print(np.isclose(self.curr_w, self.curr_w[0]).all())
        self.history.append(losses)
        return sum(losses)/len(losses)

    def compute_grads(self):
        for i in range(len(self.workers)):
            worker_out = self.workers[i].compute_grad(self.curr_w[i])
            _, self.curr_g[i], self.curr_numsam[i] = worker_out
        return self.curr_g, self.curr_numsam

    def get_avg_grad(self):
        return np.mean(self.core_opt.comb(*self.compute_grads()), axis=0)

    def step(self, lrate):
        self.core_opt.apply(self.curr_w, *self.compute_grads(), lrate)


class SchemeVar:
    def __init__(self, schemes, dim_w, num_var_samples):
        self.schemes = schemes
        self.num_var_samples = num_var_samples
        self.arrays = [np.zeros([num_var_samples,dim_w]) for scheme in schemes]

    def evaluate(self, prep_stragglers):
        for i in range(self.num_var_samples):
            prep_stragglers()
            for arr,scheme in zip(self.arrays, self.schemes):
                arr[i,:] = scheme.get_avg_grad()

        def eval_var(arr):
            diffs = arr-np.mean(arr,axis=0,keepdims=True)
            return np.mean(np.sum(diffs**2, axis=1))

        values = [eval_var(arr) for arr in self.arrays]
        for scheme,val in zip(self.schemes,values): scheme.var_history.append(val)
        return values


grad_combine_schemes = {'Equal':grad_combine_equal, 'Proportional':grad_combine_conf}


def main():
    extra = '' if _a.extra is None else '__%s'%_a.extra
    run_id = f'run_{_a.dataset}_{_a.func}_{_a.strategy}_{_a.opt}_{_a.consensus}_{_a.graph_def}_{_a.strag_dist}_{_a.strag_dist_param:g}_{_a.num_samples}_{_a.num_consensus_rounds}_{_a.doubly_stoch}_{_a.lrate_start:g}_{_a.lrate_end:g}{extra}'
    print('run_id:', run_id)
    if not os.path.exists(_a.data_dir): os.makedirs(_a.data_dir)

    dataset = model.datasets.get(_a.dataset)
    eval = model.funcs.get(_a.func)()
    Q_local_list, Q_global = strategy.reg.get(_a.strategy)(dataset)
    workers = [Worker(eval, Q_local) for Q_local in Q_local_list]
    numw = len(workers)

    if _a.consensus=='perfect':
        # simple averaging matrix
        mat_P = np.ones([numw, numw])/numw
    elif _a.consensus=='rand_walk':
        # double stochastic matrix
        W_ = make_doubly_stoch(graph_defs[_a.graph_def], _a.doubly_stoch)
        mat_P = np.linalg.matrix_power(W_, _a.num_consensus_rounds)
        print('Largest 2 eigenvalues:', eig_vals(W_)[:2])
        assert(numw==len(W_))


    dim_w = eval.get_size()
    # w_init = np.random.RandomState(seed=_a.weights_seed).normal(scale=_a.weights_scale, size=dim_w)
    w_init = eval.get_vars()
    sc = lambda comb: Scheme(workers, dim_w, Q_global,
                        opts.get(_a.opt)(w_init, mat_P, comb).init())
    schemes = [sc(grad_combine_schemes[name]) for name in _a.grad_combine]

    def prep_stragglers():
        ## set number of samples each worker is processing
        for i in range(numw):
            if _a.strag_dist=='round':
                numsam = _a.num_samples if t%numw==i else 1
            elif _a.strag_dist=='equal':
                numsam = _a.num_samples
            else:
                numsam = -1
            workers[i].prep_straggler(numsam)

    schemevar = SchemeVar(schemes, dim_w, _a.num_var_samples) if _a.eval_grad_var else None
    name_zip = lambda ll: zip(_a.grad_combine,ll)

    dd = vars(_a)
    dd['num_workers'] = numw
    dd['graph_adja_mat'] = graph_defs[_a.graph_def].tolist() if _a.graph_def else None
    dd['data'] = {name:scheme.history for name,scheme in name_zip(schemes)}
    dd['variance'] = {name:scheme.var_history for name,scheme in name_zip(schemes)}

    for t in range(_a.num_iters):
        if schemevar and t%_a.var_eval_freq==0:
            values = schemevar.evaluate(prep_stragglers)
            print('schemevar:', {name:var_ for name,var_ in name_zip(values)})

        prep_stragglers()
        lrate = _a.lrate_start - (_a.lrate_start-_a.lrate_end)*t/_a.num_iters
        for scheme in schemes: scheme.step(lrate)
        if t%_a.loss_eval_freq==0:
            print('(%d):'%t, {name:scheme.eval_global_losses() for name,scheme in name_zip(schemes)})

        if t%_a.save_freq==0 and _a.save:
            with open(os.path.join(_a.data_dir, '%s.json'%run_id), 'w') as fp_:
                json.dump(dd, fp_, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', help='dataset name', choices=model.datasets.keys())
    parser.add_argument('--strategy', help='strategy for distributing the dataset across workers', choices=strategy.reg.keys())
    parser.add_argument('--func', help='x->y function', choices=model.funcs.keys())

    parser.add_argument('--graph_def', help='worker connectivity scheme', choices=graph_defs.keys())
    parser.add_argument('--opt', help='optimizer', choices=opts.keys())

    parser.add_argument('--consensus', default='perfect', choices=['perfect', 'rand_walk'])
    parser.add_argument('--num_consensus_rounds', help='num_consensus_rounds', type=int, default=10)
    parser.add_argument('--doubly_stoch', help='method for generating doubly stochastic matrix', default='metro', choices=['metro', 'lagra'])

    parser.add_argument('--strag_dist', help='randomness in worker num_samples', default='equal', choices=['equal', 'round', 'gauss', 'bern'])
    parser.add_argument('--strag_dist_param', help='sigma or true prob in gauss/bern', type=float, default=1.)
    parser.add_argument('--num_samples', help='num_samples in each sampling method', type=int, default=60)
    parser.add_argument('--grad_combine', help='grad_combine schemes', nargs='+', default=list(grad_combine_schemes.keys()), choices=grad_combine_schemes.keys())

    parser.add_argument('--eval_grad_var', help='compute variance of gradients', action='store_true')
    parser.add_argument('--num_var_samples', help='num. samples for variance computation', type=int, default=10000)
    parser.add_argument('--var_eval_freq', help='frequency of variance computation', type=int, default=20)
    # parser.add_argument('--weights_seed', help='seed for generating init weights', type=int)
    parser.add_argument('--weights_scale', help='std.dev for initializing normal weights', type=float, default=1.)

    parser.add_argument('--num_iters', help='total iterations count', type=int, default=1000)
    parser.add_argument('--lrate_start', help='start learning rate', type=float, default=0.1)
    parser.add_argument('--lrate_end', help='end learning rate', type=float, default=0.01)

    parser.add_argument('--data_dir', default=ut.file.resolve_data_dir('consensus'))
    parser.add_argument('--save', help='save json', action='store_true')
    parser.add_argument('--extra', help='unique string for json name', type=str)
    parser.add_argument('--save_freq', help='save frequency', type=int, default=20)
    parser.add_argument('--loss_eval_freq', help='evaluate global loss frequency', type=int, default=20)
    parser.add_argument('--max_loss_eval_size', help='batch size to evaluate global loss frequency', type=int, default=-1)

    model.bind_args(parser)
    _a = parser.parse_args()
    model.store_args(_a)
    return _a


if __name__ == '__main__':
    _a = parse_args()
    print('[Arguments]', vars(_a))
    main()
