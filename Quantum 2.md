```python
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> import scipy.sparse as sp
>>> from math import pi
>>> import scipy.sparse.linalg as linalg
>>> from sympy.functions.special.delta_functions import Heaviside
>>> %matplotlib inline
```

```python
>>> h = 1e-5
>>> nodes = 2000
>>> xmin = 0
>>> xmax = 1
>>> a = 1/(nodes-1)
>>> x = np.linspace(xmin,xmax,nodes)
>>> timesteps = 100
>>> A = sp.diags([1/(2*a**2),1j/h- 1/a**2,1/(2*a**2)],[-1, 0 ,1],shape=(nodes,nodes)).tolil()
...
>>> psi = np.zeros(shape = (nodes, timesteps+1),dtype= np.cfloat)
>>> width = 0.01
>>> p0 = 50
>>> E0 = p0**2/2
>>> V0 = E0/0.6
>>> psi[:,0] = np.exp(-((x-0.38)**2/(2*width**2)))*np.exp(1j*x*p0)
>>> psi[:,0] = psi[:,0] / np.linalg.norm(psi[:,0])
...
>>> V = np.zeros(nodes)
>>> for xx in range(nodes):
...     V[xx] = V0 + V0*(Heaviside(0.4/a-xx) - Heaviside(0.4/a+1/(a*np.sqrt(V0))-xx))
...
>>> A -= sp.diags(V,0)
>>> A[0,-1] = 1/(2*a**2)
>>> A[-1,0] = 1/(2*a**2)
>>> A = A.tocsc()
...
>>> Ainv = linalg.inv(A)
...
>>> for t in range(timesteps):
...     psi[:,t+1] = Ainv*1j/h*psi[:,t]
...     psi[:,t+1] = psi[:,t+1]/np.linalg.norm(psi[:,t+1])
```

```python
>>> plt.plot(np.absolute(psi[:,0:60:10]))
>>> plt.plot(V/V0)
>>> plt.xlim(xmin=500,xmax =1000 )
```

```python
>>> h = 1e-5
>>> nodes = 2000
>>> xmin = 0
>>> xmax = 1
>>> a = 1/(nodes-1)
>>> x = np.linspace(xmin,xmax,nodes)
>>> timesteps = 500
...
>>> psi = np.zeros(shape = (nodes, timesteps+1),dtype= np.cfloat)
>>> width = 0.01
>>> p0 = 0
>>> E0 = p0**2/2
...
>>> psi[:,0] = np.exp(-((x-0.45)**2/(2*width**2)))*np.exp(1j*x*p0)
>>> psi[:,0] = psi[:,0] / np.linalg.norm(psi[:,0])
...
>>> V0 = 20000*10
>>> V = np.zeros(nodes)
>>> for xx in range(nodes):
...     V[xx] =  V0*(Heaviside(0.35/a-xx) - Heaviside(0.55/a-xx))
...
...
>>> A = sp.diags([1/(4*a**2),1j/h- 1/(2*a**2),1/(4*a**2)],[-1, 0 ,1],shape=(nodes,nodes)).tolil()
>>> A -= 0.5*sp.diags(V,0)
>>> A[0,-1] = 1/(4*a**2)
>>> A[-1,0] = 1/(4*a**2)
>>> A = A.tocsc()
>>> Ainv = linalg.inv(A)
>>> Ainv = Ainv.todense()
...
>>> B = sp.diags([-1/(4*a**2), 1j/h + 1/(2*a**2), -1/(4*a**2)],[-1,0,1],shape=(nodes,nodes)).tolil()
>>> B += 0.5*sp.diags(V,0)
>>> B[0,-1] = -1/(4*a**2)
>>> B[-1,0] = -1/(4*a**2)
>>> B = B.todense()
...
>>> C = np.dot(Ainv,B)
...
>>> for t in range(timesteps):
...     psi[:,t+1] = np.dot(C,psi[:,t])
...     psi[:,t+1] = psi[:,t+1]/np.linalg.norm(psi[:,t+1])
```

```python
>>> plt.plot(np.absolute(psi[:,0:500:100]))
>>> plt.plot(V/abs(V0))
>>> plt.xlim(xmin=600,xmax =1250 )
```

# Crank Nicholson

```python
>>> h = 1e-5
>>> nodes = 2000
>>> xmin = 0
>>> xmax = 1
>>> a = 1/(nodes-1)
>>> x = np.linspace(xmin,xmax,nodes)
>>> timesteps = 500
...
>>> psi = np.zeros(shape = (nodes, timesteps+1),dtype= np.cfloat)
>>> width = 0.01
>>> p0 = 50
>>> E0 = p0**2/2
>>> V0 = E0/0.6
...
>>> psi[:,0] = np.exp(-((x-0.52)**2/(2*width**2)))*np.exp(1j*x*p0)
>>> psi[:,0] = psi[:,0] / np.linalg.norm(psi[:,0])
...
>>> V = np.zeros(nodes)
>>> for xx in range(nodes):
...     V[xx] =  -V0*(Heaviside(0.55/a-xx) - Heaviside(0.55/a + 1/(a*np.sqrt(V0))-xx))
...     #(Heaviside(0.4/a-xx) - Heaviside(0.4/a+1/(a*np.sqrt(V0))-xx))
...
...
>>> A = sp.diags([1/(4*a**2),1j/h- 1/(2*a**2),1/(4*a**2)],[-1, 0 ,1],shape=(nodes,nodes)).tolil()
>>> A -= 0.5*sp.diags(V,0)
>>> A[0,-1] = 1/(4*a**2)
>>> A[-1,0] = 1/(4*a**2)
>>> A = A.tocsc()
>>> Ainv = linalg.inv(A)
>>> Ainv = Ainv.todense()
...
>>> B = sp.diags([-1/(4*a**2), 1j/h + 1/(2*a**2), -1/(4*a**2)],[-1,0,1],shape=(nodes,nodes)).tolil()
>>> B += 0.5*sp.diags(V,0)
>>> B[0,-1] = -1/(4*a**2)
>>> B[-1,0] = -1/(4*a**2)
>>> B = B.todense()
...
>>> C = np.dot(Ainv,B)
...
>>> for t in range(timesteps):
...     psi[:,t+1] = np.dot(C,psi[:,t])
...     psi[:,t+1] = psi[:,t+1]/np.linalg.norm(psi[:,t+1])
```

```python
>>> plt.plot(np.absolute(psi[:,0:100:50]))
>>> plt.plot(V/abs(V0))
>>> plt.xlim(xmin=900,xmax=1250 )
```

# Crank Nicholson 2-D

```python
>>> h = 1e-5
>>> nodes = 4
>>> xmin = 0
>>> xmax = 1
>>> a = 1/(nodes-1)
>>> x = np.linspace(xmin,xmax,nodes**2)
>>> timesteps = 500
...
>>> psi = np.zeros(shape = (nodes**2, timesteps+1),dtype= np.cfloat)
>>> width = 0.01
>>> p0 = 50
>>> E0 = p0**2/2
>>> V0 = E0/0.6
...
>>> psi[:,0] = np.exp(-((x-0.52)**2/(2*width**2)))*np.exp(1j*x*p0)
>>> psi[:,0] = psi[:,0] / np.linalg.norm(psi[:,0])
...
>>> # V = np.zeros(nodes)
... # for xx in range(nodes):
... #     V[xx] =  -V0*(Heaviside(0.55/a-xx) - Heaviside(0.55/a + 1/(a*np.sqrt(V0))-xx))
... #     #(Heaviside(0.4/a-xx) - Heaviside(0.4/a+1/(a*np.sqrt(V0))-xx))
...
...
... A1D = sp.diags([1/(4*a**2),1j/h- 1/(2*a**2),1/(4*a**2)],[-1, 0 ,1],shape=(nodes,nodes))
>>> I = sp.identity(nodes)
>>> A2D = sp.kron(A1D,I) + sp.kron(I,A1D)
>>> #print(A2D)
... #A2D = A.tocsc()
... #Ainv = linalg.inv(A)
... #Ainv = Ainv.todense()
...
... B1D = sp.diags([-1/(4*a**2), 1j/h + 1/(2*a**2), -1/(4*a**2)],[-1,0,1],shape=(nodes,nodes))
>>> B2D = sp.kron(B1D,I) + sp.kron(I,B1D)
>>> #B += 0.5*sp.diags(V,0)
... #B[0,-1] = -1/(4*a**2)
... #B[-1,0] = -1/(4*a**2)
...
... # for t in range(timesteps):
... #     psi[:,t+1] = np.dot(C,psi[:,t])
... #     psi[:,t+1] = psi[:,t+1]/np.linalg.norm(psi[:,t+1])
```
