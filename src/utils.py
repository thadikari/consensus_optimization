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

def resolve_data_dir_os(proj_name):
    if os.name == 'nt': # if windows
        curr_path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(curr_path, '..', 'data', 'current')
    else:
        return resolve_data_dir(proj_name)


#https://stackoverflow.com/questions/11367736/matplotlib-consistent-font-using-latex
def mpl_init(font_size=14, legend_font_size=None):
    import matplotlib.pyplot as plt
    from cycler import cycler
    import matplotlib

    custom_cycler = (cycler(color=['r', 'b', 'g', 'y', 'k']) +
                     cycler(linestyle=['-', 'dotted', ':', '-.', '-']))
    plt.rc('axes', prop_cycle=custom_cycler)
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams.update({'font.size': font_size})
    if legend_font_size: plt.rc('legend', fontsize=legend_font_size)    # legend fontsize
    # https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot


def fmt_ax(ax, xlab, ylab, leg, grid=1):
    if leg: ax.legend(loc='best')
    if xlab: ax.set_xlabel(xlab)
    if ylab: ax.set_ylabel(ylab)
    if grid: ax.grid(alpha=0.7, linestyle='-.', linewidth=0.3)
    ax.tick_params(axis='both', labelsize=12)


def save_show_fig(args, plt, file_path):
    plt.tight_layout()
    if args.save:
        for ext in args.ext:
            plt.savefig('%s.%s'%(file_path,ext), bbox_inches='tight')
    if not args.silent: plt.show()

def bind_fig_save_args(parser):
    parser.add_argument('--silent', help='do not show plots', action='store_true')
    parser.add_argument('--save', help='save plots', action='store_true')
    exts_ = ['png', 'pdf']
    parser.add_argument('--ext', help='plot save extention', nargs='*', default=exts_, choices=exts_)
