import matplotlib.pyplot as plt
from cycler import cycler
from pathlib import Path
import numpy as np
import matplotlib
import argparse
import json
import os

import graphs
import utils


utils.mpl_init()


def main():

    file_paths = Path(_a.data_dir).glob('*.json')
    proc_stem = lambda stem: stem.replace('.','') if _a.no_dots else stem
    for fpath in file_paths:
        js = json.load(open(str(fpath)))
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

        ax.legend(loc='best')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost')
        plt.gcf().set_size_inches(6.8,3)
        plt.tight_layout()
        plt.grid(alpha=0.7, linestyle='-.', linewidth=0.3)
        ax.tick_params(axis='x', labelsize=12)
        # xlabels = [('%d'%x) + 'k' for x in ax.get_xticks()/1000]
        # ax.set_xticklabels(xlabels)
        if _a.ylog: ax.set_yscale('log')
        savefig('')

        if _a.show: plt.show()
        plt.cla()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data/current')
    parser.add_argument('--ext', help='file extension', default='png', choices=['png', 'pdf'])
    parser.add_argument('--show', help='plot at the end', action='store_true')
    parser.add_argument('--graph', help='plot at the end', action='store_true')
    parser.add_argument('--all_workers', help='plot all workers', action='store_true')
    parser.add_argument('--no_dots', help='remove . from file name', action='store_true')
    parser.add_argument('--ylog', help='log axis for y', action='store_true')
    parser.add_argument('--num_iters', help='number of iterations', type=int, default=-1)
    return parser.parse_args()


if __name__ == '__main__':
    _a = parse_args()
    print('[Arguments]', vars(_a))
    main()
