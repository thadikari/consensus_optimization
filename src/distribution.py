from collections import OrderedDict


dist_types = OrderedDict()
get_type_names = lambda: list(dist_types.keys())
get_type = lambda name: dist_types[name]()


def register_(tp):
    assert(tp.__name__ not in dist_types)
    dist_types[tp.__name__] = tp
    return tp


def test():
    print(get_type_names())
    print(get_type('QPQQ'))


import dist_mnist, dist_toy
if __name__ == '__main__': test()
