from collections import OrderedDict


class Registry:
    def __init__(self): self.dd = OrderedDict()
    def keys(self): return list(self.dd.keys())
    def get(self, key): return self.dd[key]
    def put(self, key, val):
        # print(self.keys())
        assert(key not in self.dd)
        self.dd[key] = val
    def reg(self, tp):
        self.put(tp.__name__, tp)
        return tp


def mpl_init():
    import matplotlib.pyplot as plt
    from cycler import cycler
    import matplotlib

    custom_cycler = (cycler(color=['r', 'b', 'g', 'y']) +
                     cycler(linestyle=['-', '--', ':', '-.']))
    plt.rc('axes', prop_cycle=custom_cycler)
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams.update({'font.size': 14})
