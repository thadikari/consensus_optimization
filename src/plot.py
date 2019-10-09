import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
import os

from autoscale import autoscale_y


def main():

    file_paths = Path(_a.data_dir).glob('*.json')

    for fpath in file_paths:
        data = json.load(open(str(fpath)))['data']
        ax = plt.gca()
        for scheme in data: ax.plot(data[scheme], label=scheme)
        ax.legend(loc='best')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        if _a.ylog: ax.set_yscale('log')

        get_path = lambda prf: os.path.join(str(fpath.parents[0]),
                                '%s%s.%s'%(fpath.stem,prf,_a.ext))

        plt.savefig(get_path(''), bbox_inches='tight')
        length = max(len(data[scheme]) for scheme in data)

        ax.set_xlim(int(_a.xlimper*length), length)
        autoscale_y(ax)
        plt.savefig(get_path('__%g'%_a.xlimper), bbox_inches='tight')

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
