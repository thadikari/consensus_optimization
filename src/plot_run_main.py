from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import argparse
import json
import os

import graphs
import utilities as ut
import utilities.file
import utilities.mpl as utils


utils.init()
reg = ut.Registry()
register = reg.reg

def fmt_ax(*args, **kwargs):
    utils.fmt_ax(*args, **kwargs)
    if _a.ylog: args[0].set_yscale('log')
    if _a.xlog: args[0].set_xscale('log')

data_dir = lambda: os.path.join(_a.data_dir, _a.dir_name)

def main():
    file_paths = list(Path(data_dir()).glob('*.json'))
    jss = list(json.load(open(str(fpath))) for fpath in file_paths)
    reg.get(_a.type)(file_paths,jss)


@register
def plot_var(_, jss):
    keys = ['Equal', 'Proportional']
    data = [(js['toy_sigma2'], [js['variance'][key][0] for key in keys]) for js in jss]
    data.sort(key=lambda x_: x_[0])
    variance, serieses = zip(*data)
    serieses = list(zip(*serieses))
    ax = plt.gca()
    for key,series in zip(keys,serieses): ax.plot(variance, series, label=key)
    fmt_ax(ax, r'$\sigma^2$', r'$\mathbb{V}(\bar{g})$', 1)
    utils.save_show_fig(_a, plt, os.path.join(data_dir(), _a.save_name))

@register
def plot_all(*args):
    proc_stem = lambda stem: stem.replace('.','') if _a.no_dots else stem
    for fpath,js in zip(*args):
        get_path = lambda prf=None: os.path.join(str(fpath.parents[0]),
                   '%s%s'%(proc_stem(fpath.stem),'' if prf is None else '__%s'%prf))
        savefig = lambda arg: plt.savefig(get_path(arg), bbox_inches='tight')

        if not all((kw in fpath.stem) for kw in _a.keywords): continue
        plt.gcf().canvas.set_window_title(str(fpath.stem))
        # visualizing the graph structure
        if _a.graph:
            graphs.draw(np.array(js['graph_adja_mat']))
            utils.save_show_fig(_a, plt, get_path('graph'))
            plt.clf()

        ax = plt.gca()
        data = js['data']
        freq = js['loss_eval_freq']

        def proc_data(_d):
            if _a.num_iters>0: _d = _d[:int(_a.num_iters/freq)]
            if _a.filter_sigma>0: _d = gaussian_filter1d(_d, sigma=_a.filter_sigma)
            return _d

        for scheme in sorted(data.keys()):
            series = data[scheme]
            iter_ind = proc_data(np.array(range(len(series)))*freq)
            workers = list(zip(*series))
            if len(data)>1: # multiple schemes
                line, = ax.plot(iter_ind, proc_data(workers[0]), label=scheme)
                if _a.all_workers:
                    for ss in workers[1:]:
                        ax.plot(iter_ind, proc_data(ss), color=line.get_color(),
                                    linestyle=line.get_linestyle())
            else:
                for i in range(len(workers)):
                    ax.plot(iter_ind, proc_data(workers[i]), label='Wkr%d'%i)

        fmt_ax(ax, 'Iteration', 'Cost', 1)
        plt.gcf().set_size_inches(_a.fig_size)
        plt.setp(ax.get_yminorticklabels(), visible=False)
        if _a.yticks is not None: ax.set_yticks(_a.yticks)
        if _a.ylim is not None: ax.set_ylim(_a.ylim)
        if _a.xhide:
            ax.set_xticklabels([])
            ax.set_xlabel(None)
        plt.tight_layout()
        utils.save_show_fig(_a, plt, get_path())
        plt.cla()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='plot_all', choices=reg.keys())
    parser.add_argument('--dir_name', default='', type=str)
    parser.add_argument('--data_dir', default=ut.file.resolve_data_dir_os('consensus'))
    parser.add_argument('--keywords', default=[], type=str, nargs='+')

    parser.add_argument('--graph', help='plot the graph structure', action='store_true')
    parser.add_argument('--all_workers', help='plot all workers', action='store_true')
    parser.add_argument('--num_iters', help='max iteration to plot', type=int, default=-1)
    parser.add_argument('--filter_sigma', default=0, type=float)

    parser.add_argument('--save_name', help='save name', type=str)
    parser.add_argument('--fig_size', nargs=2, type=float, default=[6.5,2.2])
    parser.add_argument('--no_dots', help='remove . from file name', action='store_true')
    parser.add_argument('--xhide', help='hide xticks and label', action='store_true')
    parser.add_argument('--xlog', help='log axis for x', action='store_true')
    parser.add_argument('--ylog', help='log axis for y', action='store_true')
    parser.add_argument('--ylim', default=None, type=float, nargs=2)
    parser.add_argument('--yticks', default=None, type=float, nargs='*')

    utils.bind_fig_save_args(parser)
    return parser.parse_args()


if __name__ == '__main__':
    _a = parse_args()
    print('[Arguments]', vars(_a))
    main()
