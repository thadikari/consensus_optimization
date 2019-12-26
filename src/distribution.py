from utils import Registry
import numpy as np


reg = Registry()
register_ = reg.reg


# for typical in-memory classification datasets like mnist
class DistClassification:
    def __init__(self, xy_): self.xy_ = xy_
    def size(self): return len(self.xy_[0])

    def summary(self):
        summ = np.unique(np.argmax(self.xy_[1], axis=1), return_counts=1)
        return dict(zip(*summ))

    def sample(self, size):
        if size>0:
            tot = len(self.xy_[0])
            inds = np.random.choice(tot, size=size)
            return [z_[inds] for z_ in self.xy_]
        else:
            return self.xy_


def test():
    print(reg.keys())
    print(reg.get('QPQQ'))


import dist_mnist, dist_toy
if __name__ == '__main__': test()
