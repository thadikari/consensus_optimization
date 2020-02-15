class Registry:
    def __init__(self): self.dd = {}
    def keys(self): return list(self.dd.keys())
    def values(self): return list(self.dd.values())
    def items(self): return self.dd.items()
    def get(self, key): return self.dd[key]
    def put(self, key, val):
        assert(key not in self.dd)
        self.dd[key] = val
    def reg(self, tp):
        self.put(tp.__name__, tp)
        return tp
