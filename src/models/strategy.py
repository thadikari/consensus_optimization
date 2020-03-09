import numpy as np

import utilities.data as du
import utilities as ut


reg = ut.Registry()
reg_stg = reg.reg


# for typical in-memory classification datasets like mnist
class Dist:
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


def prep(dataset):
    x_train, y_train = dataset()
    x_train, y_train = du.permute(x_train, y_train)
    Q_global = Dist((x_train, y_train))
    return x_train, y_train, Q_global


@reg_stg
def identical(dataset):
    x_train, y_train, Q_global = prep(dataset)
    locals = [Dist((x_train, y_train)) for _ in range(dataset.num_classes)]
    return locals, Q_global


@reg_stg
def distinct(dataset):
    x_train, y_train, Q_global = prep(dataset)
    indss = [y_train==cls for cls in range(dataset.num_classes)]
    #count = min(inds.sum() for inds in indss)
    locals = [Dist((x_train[inds], y_train[inds])) for inds in indss]
    return locals, Q_global
