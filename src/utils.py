from collections import OrderedDict


class Registry:
    def __init__(self): self.dd = OrderedDict()
    def keys(self): return list(self.dd.keys())
    def get(self, key): return self.dd[key]
    def register(self, tp):
        # print(self.keys())
        assert(tp.__name__ not in self.dd)
        self.dd[tp.__name__] = tp
        return tp
