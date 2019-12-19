import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
import matplotlib
import argparse
import os


#https://stackoverflow.com/questions/11367736/matplotlib-consistent-font-using-latex
default_cycler = (cycler(color=['r', 'b', 'g', 'y', 'k']) +
                  cycler(linestyle=['-', '--', ':', '-.', '-']))
plt.rc('lines', linewidth=2)
plt.rc('axes', prop_cycle=default_cycler)
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
# matplotlib.rcParams.update({'font.size': 14})
# ax.tick_params(axis='x', labelsize=12)


registry = {}
def register_(tp): registry[tp.__name__] = tp


class Dist:
    def __init__(self, *args):
        self.rprm, self.sam_prm, self.xlabel,\
            self.name, self.title = args

    def gen(self, shape):
        for prm in np.linspace(*self.rprm, _a.n_xpts):
            yield prm, self.func(prm, shape)


@register_
class bern(Dist):
    def __init__(self):
        super().__init__((0,1), _a.bern_sample_p, r'$p$',
            '%d_%d'%(_a.bern_max, _a.bern_min),
            r'Bernoulli: $\Pr(b_i=%d)=p, \Pr(b_i=%d)=1-p$'%\
                                (_a.bern_max, _a.bern_min))
    def func(self, prm, shape):
        dd_ = (np.random.rand(*shape)<prm).astype(int)
        dd_[dd_==1] = _a.bern_max
        dd_[dd_==0] = _a.bern_min
        return dd_


@register_
class gauss(Dist):
    def __init__(self):
        super().__init__((0,_a.gauss_max_std),
                _a.gauss_sample_std, r'Standard deviation',
                '%d_%g'%(_a.gauss_loc, _a.gauss_max_std),
                r'Gaussian: mean=%d'%_a.gauss_loc)
    def func(self, prm, shape):
        dd_ = np.random.normal(loc=_a.gauss_loc, scale=prm,
                               size=shape).astype(int)
        dd_[dd_<1] = 1
        return dd_


@register_
class exp(Dist):
    def __init__(self):
        super().__init__((0,_a.exp_max_scale),
                _a.exp_sample_scale, r'Scale',
                '%d_%g'%(_a.exp_max, _a.exp_max_scale),
                r'Exponential: max=%d'%_a.exp_max)
    def func(self, prm, shape):
        dd_ = np.random.exponential(scale=prm, size=shape).astype(int)
        dd_[:] = _a.exp_max - dd_
        dd_[dd_<1] = 1
        return dd_


@register_
def mixg(shape):
    title = r'Gaussian mixture: mean=%s, stddev=%s'%(_a.mixg_loc, _a.mixg_std)
    xlabel = r'Mixture ratio'

    def func(prm):
        dd_ = np.zeros(list(shape) + [3])
        for i,(loc,std) in enumerate(zip(_a.mixg_loc, _a.mixg_std)):
            dd_[:,:,i] = np.random.normal(loc=loc, scale=std, size=shape).astype(int)
        # dd_[:] = np.min(_a.mixg_max, np.max(1,dd_[:]))
        dd_[dd_ > _a.mixg_max] = _a.mixg_max
        dd_[dd_ < 1] = 1
        sel = np.random.choice(list(range(dd_.shape[3])), size=None, replace=True, p=None)
        # print(dd_)
        return dd_[:,:,]

    def ff_():
        for prm in np.linspace(0,_a.gauss_max_std,_a.n_xpts):
            yield prm, func(prm)

    return ff_(), '%d_%d'%(_a.gauss_loc, _a.gauss_max_std),\
           title, xlabel, func, 2323



e_ = lambda v: sum(v)/len(v)
v_ = lambda v: e_((v-e_(v))**2)
lmth = lambda s_: r'$%s$'%s_


def cond():
    n = _a.n_wkr
    shape = (_a.trials, n)
    dst = registry[_a.dist]()
    bis = dst.func(dst.sam_prm, shape)
    b = bis.sum(axis=1)

    data = []
    for i in range(n,n*100):
        curr = bis[b==i]
        # print(i, curr)
        if len(curr)>0:
            b1 = curr[:,0]
            b2 = curr[:,1]
            # print(i, e_(b1*b2))
            ll = (i/n)**2, e_(b1**2), e_(b1*b2)
            data.append((i, *ll))

    le_ = lambda nu: lmth(r'\mathbb{\rm E}\left[%s\mid b \right]'%nu)
    lbls = [lmth('(b/n)^2'), le_('b_i^2'), le_('b_i b_j')]
    xv, *srs = list(zip(*data))
    for sr,lbl in zip(srs,lbls):
        # print(xv, sr)
        plt.plot(xv, sr, label=lbl)
    plt.xlabel(lmth('b'))
    plt.legend(loc='best')
    plt.show()


def main():
    n = _a.n_wkr
    shape = (_a.trials, n)
    dst = registry[_a.dist]()

    data_dir = os.path.join(os.path.dirname(__file__),'..','data', _a.data_dir)
    save = lambda sfx: plt.savefig(fname(sfx), bbox_inches='tight')
    fname = lambda sfx: os.path.join(data_dir, '%s__%s%s.%s'%(_a.dist,dst.name,sfx,_a.ext))
    plt.gcf().set_size_inches(_a.fig_size)

    # plot histogram for sample data
    sample = dst.func(dst.sam_prm, shape).flatten()
    bins = np.arange(min(sample)-1, max(sample) + 1, 1) + 0.5
    # print(bins)
    plt.hist(sample, bins=bins)
    stitle = r'Histogram of $b_i$, %s=%g'%(dst.xlabel,dst.sam_prm)
    if not _a.notitle: plt.title(stitle)
    print(stitle)
    plt.xlabel(lmth('b_i'))
    plt.ylabel('Frequency')
    plt.gca().set_yticks([])
    if _a.save: save('_hist_%g'%dst.sam_prm)
    # plt.show()
    # exit()
    plt.clf()


    # compute expected vals
    xvals, data = [], []
    for xval,bis in dst.gen(shape):
        xvals.append(xval)
        b = bis.sum(axis=1)
        b1 = bis[:,0]
        b2 = bis[:,1]

        mu2 = e_(1/b1)
        n2mu3 = e_(b1*((n/b)**2))
        # mu4 = e_((b1/b)**2)
        # mu5 = e_((b1*b2)/(b**2))
        va4 = v_(n*b1/b)
        dtild = (mu2-n2mu3)/va4
        exp = (mu2, n2mu3, n2mu3+ _a.scal*va4)#, va4, dtild)
        data.append(exp)

    le_ = lambda nu,dn,OP='E': r'\mathbb{\rm %s}\left[\frac{%s}{%s}\right]'%(OP,nu,dn)
    lf_ = lambda n_,mu,nu,dn: lmth('%s\mu_%d = %s%s'%(n_,mu,n_,le_(nu,dn)))
    l_mu2 = lf_('',2, '1','b_i')
    l_n2mu3 = lf_('n^2',3, 'b_i','b^2')
    # l_mu4 = lf_('n',4, 'b_i^2','b^2')
    # l_mu5 = lf_('(n^2-n)',5, 'b_i b_j','b^2')
    l_va4 = lmth(le_('nb_i','b','Var'))
    labels = (l_mu2, l_n2mu3, lmth('n^2 \mu_3+ %g'%_a.scal)+l_va4, l_va4, 'dtild')

    data = np.array(data).T
    for sr,label in zip(data, labels):
        # print(sr,label)
        plt.plot(xvals, sr, label=label)

    #plt.gca().set_aspect('equal', adjustable='box')
    if not _a.notitle: plt.title(dst.title)
    print(dst.title)
    plt.xlabel(dst.xlabel)
    plt.gca().set_yticks([])
    plt.legend(loc='best')
    if _a.ylog: plt.gca().set_yscale('log')
    if _a.save: save('')
    if not _a.noshow: plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dist', help='type of distribution', choices=list(registry.keys()))
    parser.add_argument('--trials', help='number of trials for monte-carlo', type=int, default=100)
    parser.add_argument('--n_wkr', help='number of workers', type=int, default=10)
    parser.add_argument('--n_xpts', help='x-axis granularity', type=int, default=30)
    parser.add_argument('--scal', help='scalar multiplier for comparison', type=float, default=0.01)


    parser.add_argument('--gauss_loc', type=int, default=60)
    parser.add_argument('--gauss_max_std', type=float, default=60)
    parser.add_argument('--gauss_sample_std', type=float, default=30)

    parser.add_argument('--bern_max', type=int, default=60)
    parser.add_argument('--bern_min', type=int, default=1)
    parser.add_argument('--bern_sample_p', type=float, default=0.8)

    parser.add_argument('--exp_max', type=int, default=60)
    parser.add_argument('--exp_max_scale', type=float, default=20)
    parser.add_argument('--exp_sample_scale', type=float, default=10)

    parser.add_argument('--mixg_max', type=int, default=250)
    parser.add_argument('--mixg_loc', type=int, nargs='+', default=[250, 170, 50])
    parser.add_argument('--mixg_std', type=float, nargs='+', default=[20, 25, 30])


    parser.add_argument('--fig_size', help='width, height', default=[4,3])
    parser.add_argument('--noshow', help='do not show plots', action='store_true')
    parser.add_argument('--notitle', help='do not show title', action='store_true')
    parser.add_argument('--yticks', help='show yticks', action='store_true')
    parser.add_argument('--ylog', help='log axis for y', action='store_true')
    parser.add_argument('--save', help='save plots', action='store_true')
    parser.add_argument('--ext', help='extention', default='png', choices=['png', 'pdf'])
    parser.add_argument('--data_dir', default='current')
    return parser.parse_args()


if __name__ == '__main__':
    _a = parse_args()
    print('[Arguments]', vars(_a))
    main()
