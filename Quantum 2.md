```python
>>> import numpy as np
>>> from numpy.fft import fft
>>> from numpy.fft import ifft
>>> import matplotlib.pyplot as plt
>>> import scipy.sparse as sp
>>> from math import pi
>>> import math
>>> import scipy.sparse.linalg as linalg
>>> from sympy.functions.special.delta_functions import Heaviside
>>> from mpl_toolkits.mplot3d import Axes3D
>>> from matplotlib import animation
...
...
>>> #%matplotlib inline
```

```python
>>> h = 1e-5
>>> nodes = 2000
>>> xmin = 0
>>> xmax = 1
>>> a = 1/(nodes - 1)
>>> x = np.linspace(xmin, xmax, nodes)
>>> timesteps = 100
>>> A = sp.diags([1/(2*a**2), 1j/h- 1/a**2,1/(2*a**2)], [-1, 0 ,1], shape = (nodes, nodes)).tolil()
...
>>> psi = np.zeros(shape = (nodes, timesteps+1), dtype= np.cfloat)
>>> width = 0.01
>>> p0 = 50
>>> E0 = p0**2 / 2
>>> V0 = E0 / 0.6
>>> psi[:, 0] = np.exp(-((x - 0.38)**2 / (2 * width**2)))*np.exp(1j*x*p0)
>>> psi[:, 0] = psi[:, 0] / np.linalg.norm(psi[:, 0])
...
>>> V = np.zeros(nodes)
>>> for xx in range(nodes):
...     V[xx] = V0 + V0*(Heaviside(0.4/a - xx) - Heaviside(0.4/a + 1/(a*np.sqrt(V0)) - xx))
...
>>> A -= sp.diags(V, 0)
>>> A[0, -1] = 1 / (2*a**2)
>>> A[-1, 0] = 1 / (2*a**2)
>>> A = A.tocsc()
...
>>> Ainv = linalg.inv(A)
...
>>> for t in range(timesteps):
...     psi[:, t+1] = Ainv * 1j/h * psi[:, t]
...     psi[:, t+1] = psi[:, t+1] / np.linalg.norm(psi[:, t+1])
```

```python
>>> plt.plot(np.absolute(psi[:, 0:60:10]))
>>> plt.plot(V / V0)
>>> plt.xlim(xmin = 500, xmax =1000)
```

# Crank Nicholson for square well

```python
>>> # Set parameters
... h = 1e-5 #timestep width
>>> nodes = 2**8
>>> xmin = 0
>>> xmax = 1
>>> a = 1 / (nodes-1) #space between nodes
>>> x = np.linspace(xmin, xmax, nodes)
>>> timesteps = 10000
...
>>> #initial wave function
... psi = np.zeros(shape = (nodes, timesteps+1), dtype= np.cfloat)
>>> width = 0.01
>>> p0 = 0 #standing still
>>> E0 = p0**2 / 2
>>> psi[:, 0] = np.exp(-((x-0.45)**2 / (2*width**2)))*np.exp(1j*x*p0)
>>> psi[:, 0] = psi[:, 0] / np.linalg.norm(psi[:, 0])
...
>>> #Make square well
... V0 = 200000
>>> V = np.zeros(nodes)
>>> for xx in range(nodes):
...     V[xx] =  V0*(Heaviside(0.35/a - xx) - Heaviside(0.55/a - xx))
...
>>> #Make left hand side matrix
... A = sp.diags([1 / (4*a**2), 1j/h - 1/(2*a**2), 1 / (4*a**2)],[-1, 0 ,1],shape=(nodes, nodes)).tolil()
>>> A -= 0.5*sp.diags(V, 0)
>>> A[0, -1] = 1 / (4*a**2)
>>> A[-1, 0] = 1 / (4*a**2)
>>> A = A.tocsc()
>>> Ainv = linalg.inv(A)
>>> Ainv = Ainv.todense()
...
>>> #make right hand side vector
... B = sp.diags([-1 / (4*a**2), 1j/h + 1/(2*a**2), -1 / (4*a**2)],[-1, 0, 1],shape=(nodes, nodes)).tolil()
>>> B += 0.5*sp.diags(V, 0)
>>> B[0, -1] = -1 / (4*a**2)
>>> B[-1, 0] = -1 / (4*a**2)
>>> B = B.todense()
...
>>> C = np.dot(Ainv,B)
...
>>> for t in range(timesteps):
...     psi[:, t+1] = np.dot(C, psi[:, t])
...     psi[:, t+1] = psi[:, t+1] / np.linalg.norm(psi[:, t+1])
```

```python
>>> plt.plot(np.absolute(psi[:,0:500:100]))
>>> plt.plot(V/abs(V0)) #plot normalized potential well
>>> plt.xlim(xmin=0,xmax =2**8 )
```

```python
>>> fig = plt.figure()
>>> ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
>>> line, = ax.plot([], [], lw=2)
...
...
>>> # initialization function: plot the background of each frame
... def init():
...     line.set_data([], [])
...     return line,
...
...
>>> # animation function.  This is called sequentially
... def animate(i):
...     y = np.absolute(psi[:,i])
...     line.set_data(x, y)
...     return line,
...
>>> # call the animator.  blit=True means only re-draw the parts that have changed.
... anim = animation.FuncAnimation(fig, animate, init_func=init,
...            frames=10000, interval=20, blit=True)
...
>>> plt.show()
```

# Crank Nicholson for tunneling

```python
>>> #Set up parameters
... h = 1e-5
>>> nodes = 2000
>>> xmin = 0
>>> xmax = 1
>>> a = 1 / (nodes-1)
>>> x = np.linspace(xmin,xmax,nodes)
>>> timesteps = 10000
...
>>> #Set up wave funciton
... psi = np.zeros(shape = (nodes, timesteps+1), dtype= np.cfloat)
>>> width = 0.01
>>> p0 = 50
>>> psi[:, 0] = np.exp(-((x - 0.52)**2 / (2*width**2)))*np.exp(1j*x*p0)
>>> psi[:, 0] = psi[:, 0] / np.linalg.norm(psi[:, 0])
...
>>> #Set up potential wall with height E0/0.6
... E0 = p0**2 / 2
>>> V0 = E0 / 0.6
>>> V = np.zeros(nodes)
>>> for xx in range(nodes):
...     V[xx] =  -V0*(Heaviside(0.55/a - xx) - Heaviside(0.55/a + 1/(a*np.sqrt(V0)) - xx))
...
>>> #Create left hand side matrix
... A = sp.diags([1 / (4*a**2), 1j/h- 1/(2*a**2), 1 / (4*a**2)],[-1, 0 ,1],shape=(nodes, nodes)).tolil()
>>> A -= 0.5 * sp.diags(V, 0)
>>> A[0, -1] = 1 / (4*a**2)
>>> A[-1, 0] = 1 / (4*a**2)
>>> A = A.tocsc()
>>> Ainv = linalg.inv(A)
>>> Ainv = Ainv.todense()
...
>>> #Create right hand side matrix
... B = sp.diags([-1 / (4*a**2), 1j/h + 1/(2*a**2), -1 / (4*a**2)],[-1, 0, 1], shape = (nodes, nodes)).tolil()
>>> B += 0.5 * sp.diags(V, 0)
>>> B[0, -1] = -1 / (4*a**2)
>>> B[-1, 0] = -1 / (4*a**2)
>>> B = B.todense()
...
>>> C = np.dot(Ainv,B)
...
>>> for t in range(timesteps):
...     psi[:, t+1] = np.dot(C, psi[:, t])
...     psi[:, t+1] = psi[:, t+1] / np.linalg.norm(psi[:, t+1])
```

```python
>>> plt.plot(np.absolute(psi[:, 0:100:50]))
>>> plt.plot(V / abs(V0))
>>> plt.xlim(xmin=900, xmax=1250)
```

```python
>>> fig = plt.figure()
>>> ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
>>> line, = ax.plot([], [], lw=2)
...
...
>>> # initialization function: plot the background of each frame
... def init():
...     line.set_data([], [])
...     return line,
...
...
>>> # animation function.  This is called sequentially
... def animate(i):
...     y = np.absolute(psi[:,i])
...     line.set_data(x, y)
...     return line,
...
>>> # call the animator.  blit=True means only re-draw the parts that have changed.
... anim = animation.FuncAnimation(fig, animate, init_func=init,
...            frames=10000, interval=20, blit=True)
...
>>> plt.show()
```

# Crank Nicholson 2-D

```python
>>> #Set up parameters
... h = 1e-5
>>> nodes = 2**8
>>> xmin = 0
>>> xmax = 1
>>> a = 1 / (nodes-1)
>>> x = np.linspace(xmin, xmax, nodes)
>>> timesteps = 200
...
>>> #Make initial wave function using x-lexicographic ordering of gridpoints
... psi = np.zeros(shape = (nodes, 1), dtype= np.cfloat)
>>> width = 0.01
>>> p0 = 50
>>> E0 = p0**2 / 2
>>> V0 = E0 / 0.6
>>> psi = np.exp(-((x-0.1)**2 / (2*width**2))) * np.exp(1j*x*p0)
>>> psi = psi / np.linalg.norm(psi)
>>> psi2 = np.tile(psi, (nodes, 1)) #Make a 2d vector by stacking Gaussians
>>> psi2 = psi2.reshape(-1) #Make into 1d vector
...
>>> #Make Potential barrier
... V = np.zeros((nodes, nodes))
>>> V0 = 1e6
>>> aa = 5
>>> bb = 40
>>> middle = int(nodes / 2)
>>> wall = int(nodes/5)
>>> for xx in range(nodes):
...     V[xx,wall] = V0*(math.ceil((Heaviside(xx)-Heaviside(xx-(middle-bb-0.5*aa)))+(Heaviside(xx-(middle-0.5*aa))\
...                                             -Heaviside(xx-(middle+0.5*aa)))+Heaviside(xx-(middle+0.5*aa+bb))))
...
...
>>> #Make left hand side vector by expanding the 1D case (using dirichlet boundary conditions at 0)
... A1D = sp.diags([1/(4*a**2), 1j/h - 1/(2*a**2), 1 / (4*a**2)], [-1, 0 ,1], shape = (nodes, nodes))
>>> I = sp.identity(nodes)
>>> A2D = sp.kron(A1D, I) + sp.kron(I, A1D)
>>> A2D -= 0.5 * sp.diags(V.reshape(-1), 0)
...
>>> #Make right hand side vector in the same way
... B1D = sp.diags([-1/(4*a**2), 1j/h + 1/(2*a**2), -1/(4*a**2)],[-1, 0, 1], shape = (nodes, nodes))
>>> B2D = sp.kron(B1D, I) + sp.kron(I, B1D)
>>> B2D += 0.5 * sp.diags(V.reshape(-1), 0)
...
>>> #Using a bicgstab algorithm, solve A*psi(n+1) = B*psi(n) for psi(n+1)
... psi2_old = psi2
>>> psi2_new = psi2
>>> for t in range(timesteps):
...     b = B2D.dot(psi2_old)
...     psi2_new = linalg.bicgstab(A2D, b)[0]
...     psi2_new = psi2_new / np.linalg.norm(psi2_new)
...     psi2_old = psi2_new
...
>>> psi2_plot = psi2_new.reshape(nodes, -1) #reshape for plotting
```

```python
>>> xx = range(nodes)
>>> hf = plt.figure()
>>> X, Y = np.meshgrid(xx, xx)
>>> plt.contour(X[:,wall:], Y[:,wall:], np.absolute(psi2_plot[:,wall:]))
>>> plt.show()
```

```python

```
