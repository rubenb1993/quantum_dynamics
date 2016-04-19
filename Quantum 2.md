```python
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> import scipy.sparse as sp
>>> import scipy.sparse.linalg as linalg
>>> from sympy.functions.special.delta_functions import Heaviside
>>> %matplotlib inline
```

```python
>>> h = 1e-5
>>> nodes = 1000
>>> a = 1/(nodes-1)
>>> x = np.linspace(0,nodes-1,nodes)
>>> timesteps = 1000
>>> A = sp.diags([1/(2*a**2),1j/h- 1/a**2,1/(2*a**2)],[-1, 0 ,1],shape=(nodes,nodes)).tocsc()
...
>>> V = np.zeros(nodes)
>>> for xx in range(nodes):
...     V[xx] = 10 + 30*(Heaviside(300-xx) - Heaviside(400-xx))
...
>>> A -= sp.diags(V,0)
>>> A = A.tocsc()
>>> psi = np.zeros(shape = (nodes, timesteps+1),dtype= np.cfloat)
>>> psi[:,0] = np.exp(-((x-200)/10)**2)*np.exp(1j*x*0.1)
...
>>> Ainv = linalg.inv(A)
...
>>> for t in range(timesteps):
...     psi[:,t+1] = Ainv*1j/h*psi[:,t]
```

```python
>>> plt.plot(np.absolute(psi[:,0:200:40]))
>>> #plt.plot(V)
```
