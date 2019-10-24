import numpy as np

count, n = 10000, 1000
# bi_smpl = lambda shp: np.random.normal(loc=100.0, scale=50.0, size=shp)
bi_smpl = lambda shp: (np.random.rand(*shp)>0.8)*64
# bi_smpl = lambda shp: np.ones(shp)
bis = bi_smpl((count, n)).astype(int)
bis[bis<1] = 1


b = bis.sum(axis=1)
b1 = bis[:,0]
b2 = bis[:,1]

rv = {'nbi__b'      : (n*b1)/b,
      'inv__bi'     : 1/b1,
      'nsqbi__bsq'  : b1*((n/b)**2),
      'nsqbisq__bsq': ((n*b1)/b)**2,
      'nbibj__bsq'  : (n*b1*b2)/(b**2)}

exp = lambda v_: sum(v_)/len(v_)
var = lambda v_: exp((v_-exp(v_))**2)

for key in rv:
    print('E[%s]:'%key, exp(rv[key]))
