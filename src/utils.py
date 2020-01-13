from collections import OrderedDict
import os


class Registry:
    def __init__(self): self.dd = OrderedDict()
    def keys(self): return list(self.dd.keys())
    def values(self): return list(self.dd.values())
    def items(self): return self.dd.items()
    def get(self, key): return self.dd[key]
    def put(self, key, val):
        # print(self.keys())
        assert(key not in self.dd)
        self.dd[key] = val
    def reg(self, tp):
        self.put(tp.__name__, tp)
        return tp


def resolve_data_dir(proj_name):
    SCRATCH = os.environ.get('SCRATCH', None)
    if not SCRATCH: SCRATCH = os.path.join(os.path.expanduser('~'), 'SCRATCH')
    return os.path.join(SCRATCH, proj_name)


#https://stackoverflow.com/questions/11367736/matplotlib-consistent-font-using-latex
def mpl_init():
    import matplotlib.pyplot as plt
    from cycler import cycler
    import matplotlib

    custom_cycler = (cycler(color=['r', 'b', 'g', 'y', 'k']) +
                     cycler(linestyle=['-', '--', ':', '-.', '-']))
    plt.rc('axes', prop_cycle=custom_cycler)
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams.update({'font.size': 14})


def fmt_ax(ax, xlab, ylab, leg, grid=1):
    if leg: ax.legend(loc='best')
    if xlab: ax.set_xlabel(xlab)
    if ylab: ax.set_ylabel(ylab)
    if grid: ax.grid(alpha=0.7, linestyle='-.', linewidth=0.3)
    ax.tick_params(axis='both', labelsize=12)
