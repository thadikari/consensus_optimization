import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx
import numpy as np
import argparse
import json
import os

from autoscale import autoscale_y


def main():

    file_paths = Path(_a.data_dir).glob('*.json')

    for fpath in file_paths:
        js = json.load(open(str(fpath)))
        get_path = lambda prf: os.path.join(str(fpath.parents[0]),
                                '%s%s.%s'%(fpath.stem,prf,_a.ext))
        savefig = lambda arg: plt.savefig(get_path(arg), bbox_inches='tight')

        adja_mat = np.array(js['graph_adja_mat'])
        G = nx.from_numpy_matrix(adja_mat)
        vrts = list(range(len(adja_mat)))
        labels = {i:str(i) for i in vrts}
        # nx.draw_spring(G, labels=labels)
        pos = dict(zip(vrts,np.array([vrts, np.array(vrts)+10]).T))
        nx.draw(G, nx.spring_layout(G, pos=pos), labels=labels)
        plt.axis('equal')
        savefig('__graph')
        if _a.show: plt.show()
        plt.clf()

        ax = plt.gca()
        data = js['data']
        freq = js['loss_eval_freq']
        for scheme in data:
            series = data[scheme]
            iter_ind = np.array(range(len(series)))*freq
            workers = list(zip(*series))
            if len(data)>1: # multiple schemes
                line, = ax.plot(iter_ind, workers[0], label=scheme)
                for ss in workers[1:]: ax.plot(iter_ind, ss, color=line.get_color())
            else:
                for i in range(len(workers)):
                    ax.plot(iter_ind, workers[i], label='Wkr%d'%i)

        ax.legend(loc='best')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        if _a.ylog: ax.set_yscale('log')

        savefig('')
        length = max(len(data[scheme])*freq for scheme in data)

        ax.set_xlim(int(_a.xlimper*length), length)
        autoscale_y(ax)
        savefig('__%g'%_a.xlimper)

        if _a.show: plt.show()
        plt.cla()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data/current')
    parser.add_argument('--ext', help='file extension', default='png', choices=['png', 'pdf'])
    parser.add_argument('--show', help='plot at the end', action='store_true')
    parser.add_argument('--ylog', help='log axis for y', action='store_true')
    parser.add_argument('--xlimper', help='xlim starting percentage', type=float, default=0.5)
    return parser.parse_args()


if __name__ == '__main__':
    _a = parse_args()
    print('[Arguments]', vars(_a))
    main()
