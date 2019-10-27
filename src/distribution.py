from utils import Registry


reg = Registry()
register_ = reg.register


def test():
    print(reg.keys())
    print(reg.get('QPQQ'))


import dist_mnist, dist_toy
if __name__ == '__main__': test()
