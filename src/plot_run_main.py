import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import argparse
import json
import os

import graphs
import utils


utils.mpl_init()
reg = utils.Registry()
register = reg.reg

def fmt_ax(*args, **kwargs):
    utils.fmt_ax(*args, **kwargs)
    if _a.ylog: args[0].set_yscale('log')
    if _a.xlog: args[0].set_xscale('log')


def main():
    file_paths = list(Path(_a.data_dir).glob('*.json'))
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
    path = os.path.join(_a.data_dir, '%s.%s'%(_a.name,_a.ext))
    plt.savefig(path, bbox_inches='tight')
    if _a.show: plt.show()

@register
def plot_all(*args):
    proc_stem = lambda stem: stem.replace('.','') if _a.no_dots else stem
    for fpath,js in zip(*args):
        get_path = lambda prf: os.path.join(str(fpath.parents[0]),
                        '%s%s.%s'%(proc_stem(fpath.stem),prf,_a.ext))
        savefig = lambda arg: plt.savefig(get_path(arg), bbox_inches='tight')

        # visualizing the graph structure
        if _a.graph:
            graphs.draw(np.array(js['graph_adja_mat']))
            savefig('__graph')
            if _a.show: plt.show()
            plt.clf()

        ax = plt.gca()
        data = js['data']
        freq = js['loss_eval_freq']
        for scheme in sorted(data.keys()):
            series = data[scheme]
            proc_data = lambda _d: _d[:int(_a.num_iters/freq)] if _a.num_iters>0 else _d
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
        plt.gcf().set_size_inches(6.8,3)
        plt.tight_layout()
        # xlabels = [('%d'%x) + 'k' for x in ax.get_xticks()/1000]
        # ax.set_xticklabels(xlabels)
        savefig('')

        if _a.show: plt.show()
        plt.cla()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='plot_all', choices=reg.keys())
    parser.add_argument('--data_dir', default=os.path.join('..','data','current'))
    parser.add_argument('--ext', help='file extension', default='png', choices=['png', 'pdf'])
    parser.add_argument('--name', help='save name', type=str)
    parser.add_argument('--show', help='plot at the end', action='store_true')
    parser.add_argument('--graph', help='plot at the end', action='store_true')
    parser.add_argument('--all_workers', help='plot all workers', action='store_true')
    parser.add_argument('--no_dots', help='remove . from file name', action='store_true')
    parser.add_argument('--xlog', help='log axis for x', action='store_true')
    parser.add_argument('--ylog', help='log axis for y', action='store_true')
    parser.add_argument('--num_iters', help='number of iterations', type=int, default=-1)
    return parser.parse_args()


if __name__ == '__main__':
    _a = parse_args()
    print('[Arguments]', vars(_a))
    main()
