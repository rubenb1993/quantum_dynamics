# Crank-Nicholson for Quantum Dynamics

Code by Ruben Biesheuvel and Alexander Harms

This code uses a Crank-Nicholson method to evaluate the time evaluation of a wave function governed by the Schrödinger equation.

```python
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> import scipy.sparse as sp
>>> import matplotlib #sometimes animation does not work, hence also matplotlib is imported
>>> from math import pi
>>> import math
>>> import scipy.sparse.linalg as linalg
>>> from sympy.functions.special.delta_functions import Heaviside
>>> from matplotlib import animation
>>> import matplotlib.cm as cm
>>> from IPython.display import Image
>>> from matplotlib import rc
...
>>> # Define font for figures
... rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
>>> rc('text', usetex=True)
...
>>> #%matplotlib inline
```

```python
>>> #Functions for 1D and 2D animations
...
... # initialization function: plot the background of each frame
... def init_1D():
...     "Initializes the 1d animation with a clean slate"
...     line.set_data([], [])
...     V_x.set_data([], [])
...     return line,V_x,
...
...
>>> # animation function.  This is called sequentially for every frame.
... def animate_1D(i):
...     "sets the frames for the animation in 1d"
...     y = np.absolute(psi_animate[:, i])
...     line.set_data(x, y)
...     V_x.set_data(x, V)
...     return line, V_x,
...
...
>>> def animation_1D():
...     "calls the animation function in 1d"
...     anim = animation.FuncAnimation(fig, animate_1D, init_func=init_1D,
...            frames=animation_frames, save_count=anim_count, interval=20, blit=False, repeat=False)
...     return anim
...
>>> #Functions for 2D animations
... def init_without_wall():
...     "initializes the animation with a clean slate"
...     cont = ax.contourf(X_wall, Y_wall, Z, cmap=cm.bone)
...     cbar = plt.colorbar(cont)
...     return cont
...
...
>>> def animate_without_wall(t):
...     "sets the frames for the animation with no wall"
...     Z = Z_wall[:, :, t]
...     cont = ax.contourf(X_wall, Y_wall, Z, cmap=cm.bone)
...     return cont
...
...
>>> def animation_2d_no_wall():
...     "calls the animation function in 2d for no wall"
...     anim2D = matplotlib.animation.FuncAnimation(fig, animate_without_wall, frames=animation_frames,
...              save_count=anim_count, interval=1, repeat=False,  init_func=init_without_wall,)
...     return anim2D
...
...
>>> def init_2d():
...     "initializes the 2d plot for the whole domain with a clean slate"
...     cont = ax.contourf(X, Y, Z, cmap=cm.bone)
...     V_x = ax.contour(xx, xx, V)
...     cbar = plt.colorbar(cont)
...     return cont, V_x,
...
...
>>> def animate_2d(t):
...     "sets the frames for the animation in 2d with the whole domain"
...     Z = z[:,:,t]
...     cont = ax.contourf(X, Y, Z, cmap=cm.bone)
...     V_x = ax.contour(xx, xx, V)
...     return cont, V_x
...
...
>>> def animation_2d():
...     "calls the animation function in 2d for the whole domain"
...     anim2D = matplotlib.animation.FuncAnimation(fig, animate_2d, frames=animation_frames, save_count=anim_count,
...             interval=1, repeat=False,  init_func=init_2d,)
...     return anim2D
...
>>> def save_anim(file, title):
...     "saves the animation with a desired title"
...     Writer = animation.writers['ffmpeg']
...     writer = Writer(fps=25, metadata=dict(artist='Me'), bitrate=1800)
...     file.save(title + '.mp4', writer=writer)
```

# Crank Nicholson for square well

A stationary Guassian Wave packet is initialized in an "infinite" potential well. The time evolution of the wave equation is calculated through the Schrödinger equation. This is approximated using a Crank Nicholson method, and with continous boundary conditions (i.e. the x axis begin is "tied" to the end). The equation that is solved with the Crank Nicholson method is the following

$$ i \hbar \frac{\psi(t+h) - \psi(t)}{h} = \frac{ \hat{H} \psi(t) + \hat{H} \psi(t+h)}{2}$$,

where h is the size of the time step, and where $\hat{H}$ is defined as

$$ \hat{H} = \frac{\hat{p}^2}{2m} + \hat{V}$$.

The momentum operator can be discretized in a grid with Dirichlet boundary conditions, and with equal spacing $a$, the matrix of the Hamiltonian is:

$$ \hat{H} = \begin{bmatrix}
1/a^2 + V_0 & -1/2a^2 & 0 & \ldots  & \ldots &  0 \\
-1/2a^2 & 1/a^2 + V_1 & -1/2a^2 & 0 & \ldots & 0  \\
0 & \ddots & \ddots & \ddots & \ldots & 0 \\
0 & \ldots &  0 & -1/2a^2  & 1/a^2 + V_{n-1} & -1/2a^2 & \\
0 & 0 & \ldots & 0 & -1/2a^2&  1/a^2 + V_n \\
\end{bmatrix}
$$,

where $V_i$ is the potential at point $i$.

The system that is being solved is then expressed as

$$ \left( \frac{i \hbar}{h} - \frac{\hat{H}}{2}\right) \psi(t + h) = \left( \frac{i \hbar}{h} + \frac{\hat{H}}{2} \right) \psi(t)$$,

where $\psi(t+h)$ is the desired quantity, and the rest is known. This expression can be simplified to a linear equation as $A \mathbf{x} = \mathbf{b}$, where $A = \left( \frac{i \hbar}{h} - \frac{\hat{H}}{2}\right) $, $\mathbf{x}$ the desired quantity $\psi(t+h)$ and $\mathbf{b} = \left( \frac{i \hbar}{h} + \frac{\hat{H}}{2}\right) \psi(t)$.

Because the coefficient matrix $A$ has an imaginary diagonal, this is not a Hermitian matrix. An algorithm designed for solving such a system can used, two of which are the Bi-Conjugate Gradient Stabalized (BiCGStab) algorithm and the Complex Orthogonal Conjugate Gradient (COCG) algorithm.

In this code, the BiCGStab algorithm has been implemented due to it being readily available in the SciPy package.

```python
>>> # Set parameters
... h = 1e-5  # timestep size
>>> xmin = 0
>>> xmax = 1
>>> nodes = 2**8
...
>>> a = (xmax - xmin) / (nodes-1)  # space between nodes
>>> x = np.linspace(xmin, xmax, nodes)
>>> timesteps = 20000
...
>>> # initial wave function
... psi = np.zeros(shape=(nodes, timesteps+1), dtype=np.cfloat)
>>> width = 0.01
>>> p0 = 0  # standing still
>>> E0 = p0**2 / 2
>>> psi[:, 0] = np.exp(-((x-0.45)**2 / (2*width**2))) * np.exp(1j*x*p0)
>>> psi[:, 0] = psi[:, 0] / np.linalg.norm(psi[:, 0])
...
>>> # Make square well
... V0 = 1e6  # High number for "infinite" sq well
>>> V = np.zeros(nodes)
>>> for xx in range(nodes):
...     V[xx] =  -V0 * (Heaviside(0.35/a - xx) - Heaviside(0.55/a - xx))
...
>>> # Make left hand side matrix
... A = sp.diags([1 / (4*a**2), 1j/h - 1/(2*a**2), 1 / (4*a**2)], [-1, 0 ,1],shape=(nodes, nodes)).tolil()
>>> A -= 0.5 * sp.diags(V, 0)
>>> A = A.tocsc()
...
...
>>> # Make right hand side vector
... B = sp.diags([-1 / (4*a**2), 1j/h + 1/(2*a**2), -1 / (4*a**2)], [-1, 0, 1],shape=(nodes, nodes)).tolil()
>>> B += 0.5*sp.diags(V, 0)
>>> B = B.tocsc()
...
>>> anim_constant = 10
>>> anim_count = 0
>>> animation_frames = int((timesteps+1)/anim_constant)
...
>>> psi_animate = np.zeros(shape=(nodes,animation_frames), dtype=np.cfloat)
>>> for t in range(timesteps):
...     b = B.dot(psi[:, t])
...     psi[:, t+1] = linalg.bicgstab(A, b)[0]
...     psi[:, t+1] = psi[:, t+1] / np.linalg.norm(psi[:, t+1])
...     if (t+1) % anim_constant == 0:
...         psi_animate[:, anim_count] = psi[:, t+1]
...         anim_count += 1
```

```python
>>> # Show animations
... fig = plt.figure()
>>> ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
>>> line, = ax.plot([], [], lw=2)
>>> V_x, = ax.plot([],[], lw=2)
...
>>> # call the animator.
... anim_sq_well = animation_1D()
...
>>> plt.show()
```

```python
>>> save_anim(anim_sq_well,'square_well_20k')
```

# Crank-Nicholson for tunneling

```python
>>> # Set up parameters
... h = 1e-5
>>> nodes = 2**8
>>> xmin = 0
>>> xmax = 1
>>> a = (xmax - xmin) / (nodes-1)
>>> x = np.linspace(xmin, xmax, nodes)
>>> timesteps = 1000
...
>>> # Set up wave funciton
... psi = np.zeros(shape=(nodes, timesteps+1), dtype=np.cfloat)
>>> width = 0.01
>>> p0 = 300
>>> psi[:, 0] = np.exp(-((x - 0.4)**2 / (2*width**2))) * np.exp(1j*x*p0)
>>> psi[:, 0] = psi[:, 0] / np.linalg.norm(psi[:, 0])
...
>>> # Set up potential wall with height E0/0.6
... E0 = p0**2 / 2
>>> V0 = E0 / 0.6
>>> V = np.zeros(nodes)
>>> for xx in range(nodes):
...     V[xx] =  -V0 * (Heaviside(0.55/a - xx) - Heaviside(0.55/a + 1/(a*np.sqrt(V0)) - xx))
...
>>> # Create left hand side matrix
... A = sp.diags([1 / (4*a**2), 1j/h - 1/(2*a**2), 1 / (4*a**2)], [-1, 0 ,1],shape=(nodes, nodes)).tolil()
>>> A -= 0.5 * sp.diags(V, 0)
>>> A = A.tocsc()
...
>>> # Create right hand side matrix
... B = sp.diags([-1 / (4*a**2), 1j/h + 1/(2*a**2), -1 / (4*a**2)], [-1, 0, 1], shape=(nodes, nodes)).tolil()
>>> B += 0.5 * sp.diags(V, 0)
>>> B = B.tocsc()
...
>>> anim_constant = 2
>>> anim_count = 0
>>> animation_frames = int((timesteps+1)/anim_constant)
...
>>> psi_animate = np.zeros(shape=(nodes,animation_frames), dtype=np.cfloat)
>>> for t in range(timesteps):
...     b = B.dot(psi[:, t])
...     psi[:, t+1] = linalg.bicgstab(A, b)[0]
...     psi[:, t+1] = psi[:, t+1] / np.linalg.norm(psi[:, t+1])
...     if (t+1) % anim_constant == 0:
...         psi_animate[:, anim_count] = psi[:, t+1]
...         anim_count += 1
```

```python
>>> # Show an animation
... fig = plt.figure()
>>> ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
>>> line, = ax.plot([], [], lw=2)
>>> V_x, = ax.plot([],[], lw=2)
...
>>> animtunnel = animation_1D()
>>> plt.show()
```

```python
>>> save_anim(animtunnel,'1d_tunnel')
```

# Crank-Nicholson in two dimensions

The 2D method of Crank-Nicholson makes use of a lexicographic ordering of nodes (image taken from [1]),
<img src="lexicograph.png",width=200,height=200>.

This results in the solution vector $u \in \mathbb{R}^{nodes^2}$ and the Coefficient matrix $A \in \mathbb{R}^{(nodes^2)~ \times~ (nodes^2)}$.

For this two dimensional problem, the boundary conditions are taken to be Dirichlet boundary conditions, with the forcing term equal to 0 for simplicity. If this is done, the coefficient matrix can be expressed as [1]:

$$\hat{H}_{2D} = \hat{H}_{1D} \otimes \mathbb{1} + \mathbb{1} \otimes \hat{H}_{1D}$$,

where $H_{1D}$ is the Hamiltonian matrix for the 1D case without the potential, and $\mathbb{1}$ is the identity matrix with the same size as $H_{1D}$. The potential is removed due to the fact that it is a 2D potential that can not be expressed in a 1D case. The potential diagonal is added after the 2D kinetic energy matrix is generated.

The system that is being solved can therefore be expressed as

$$ \left( \frac{i \hbar}{h} - \frac{\hat{H}_{2D}}{2} - \frac{\hat{V}}{2} \right) \psi(t + h) = \left( \frac{i \hbar}{h} + \frac{\hat{H}_{2D}}{2} + \frac{\hat{V}}{2} \right) \psi(t)$$,

where $\psi(t+h)$ is the desired quantity, and the rest is known. This expression can be simplified to a linear equation as $A \mathbf{x} = \mathbf{b}$, where $A = \left( \frac{i \hbar}{h} - \frac{\hat{H}_{2D}}{2} - \frac{\hat{V}}{2} \right) $, $\mathbf{x}$ the desired quantity $\psi(t+h)$ and $\mathbf{b} = \left( \frac{i \hbar}{h} + \frac{\hat{H}_{2D}}{2} + \frac{\hat{V}}{2} \right) \psi(t)$.

This system is again solved with a BiCGStab algorithm.

$\large \bf{References}$

[1] C. Vuik and D.J.P. Lahaye. $\textit{Lecture notes in Scientific Computing}$. Sept. 2015.

```python
>>> # Set up parameters
... h = 1e-5
>>> nodes = 2**7
>>> xmin = 0
>>> xmax = 1
>>> a = (xmax - xmin) / (nodes-1)
>>> x = np.linspace(xmin, xmax, nodes)
>>> timesteps = 5000
>>> wall = int(2*nodes / 3)
>>> middle = int(nodes / 2)
...
>>> #Make initial wave function using x-lexicographic ordering of gridpoints, by first making a 1d gaussian
... #and then stacking it vertically
... psi = np.zeros(shape = (len(x), ), dtype= np.cfloat)
>>> width = 0.03
>>> p0 = 500
>>> E0 = p0**2 / 2
>>> wave_position = np.arange(0, wall*a - 3*width, 3*width)
>>> for i in range(len(wave_position)):
...     psi += np.exp(-((x-wave_position[i])**2 / (2*width**2))) * np.exp(1j*x*p0)
...
>>> psi = psi / np.linalg.norm(psi)
>>> psi2 = np.tile(psi, (nodes, 1)) # Make a 2d vector by stacking Gaussians
>>> psi2_plot = np.zeros((nodes, nodes, timesteps+1), dtype=np.cfloat)
>>> psi2_plot[:, :, 0] = psi2
>>> psi2 = psi2.reshape(-1)  # Make into 1d vector
...
>>> # Make potential barrier
... V = np.zeros((nodes, nodes))
>>> V0 = 1e6
>>> aa = 5 #width of barrier between the holes in nodes
>>> bb = 10 #width of the holes in nodes
>>> for xx in range(nodes):
...     V[xx,wall] = V0 * (math.ceil((Heaviside(xx) - Heaviside(xx - (middle - bb - 0.5*aa))) + (Heaviside(xx - (middle - 0.5*aa)) \
...                       - Heaviside(xx - (middle + 0.5*aa))) + Heaviside(xx - (middle + 0.5*aa + bb))))
...
>>> # Make left hand side vector by expanding the 1D case (using zero-valued Dirichlet boundary conditions)
... A1D = sp.diags([1 / (4*a**2), 1j/h - 1/(2*a**2), 1 / (4*a**2)], [-1, 0 ,1], shape=(nodes, nodes))
>>> I = sp.identity(nodes)
>>> A2D = sp.kron(A1D, I) + sp.kron(I, A1D)
>>> A2D -= 0.5 * sp.diags(V.reshape(-1), 0)
...
>>> # Make right hand side vector in the same way
... B1D = sp.diags([-1 / (4*a**2), 1j/h + 1/(2*a**2), -1 / (4*a**2)], [-1, 0, 1], shape = (nodes, nodes))
>>> B2D = sp.kron(B1D, I) + sp.kron(I, B1D)
>>> B2D += 0.5 * sp.diags(V.reshape(-1), 0)
...
>>> # Using a bicgstab algorithm, solve A*psi(n+1) = B*psi(n) for psi(n+1)
...
... anim_constant = 10
>>> anim_count = 0
>>> animation_frames = int((timesteps+1)/anim_constant)
>>> psi2_animate  = np.zeros((nodes, nodes, animation_frames), dtype=np.cfloat)
...
>>> psi2_old = psi2
>>> psi2_new = psi2
...
>>> for t in range(timesteps):
...     b = B2D.dot(psi2_old)
...     psi2_new = linalg.bicgstab(A2D, b)[0]
...     psi2_new = psi2_new / np.linalg.norm(psi2_new)
...     psi2_old = psi2_new
...     psi2_plot[:, :, t+1] = psi2_new.reshape(nodes, -1) #reshape for plotting
...     if (t+1) % anim_constant == 0:
...         psi2_animate[:, :, anim_count] = psi2_plot[:, :, t+1]
...         anim_count += 1
```

```python
>>> # Animation of only past the wall
... fig, ax = plt.subplots()
...
>>> xx = range(nodes)
>>> X,Y = np.meshgrid(xx, xx)
>>> X_wall = X[:, wall:]
>>> Y_wall = Y[:, wall:]
>>> Z = np.zeros(X_wall.shape)
>>> z = np.absolute(psi2_animate)
>>> Z_wall = z[:, wall:, :]
...
>>> interference_no_wall = animation_2d_no_wall()
>>> plt.show()
```

```python
>>> save_anim(interference_no_wall,'interference_no_wall_500p_5000t')
```

```python
>>> # Animation including everything
... fig, ax = plt.subplots()
...
>>> xx = range(nodes)
>>> X,Y = np.meshgrid(xx, xx)
>>> Z = np.zeros(X.shape)
>>> z = np.absolute(psi2_animate)
...
>>> interference_with_animation = animation_2d()
>>> plt.show()
```

```python
>>> save_anim(interference_with_animation,'interference_with_wall')
```

# Crank-Nicholson 2D Square well

```python
>>> # Set up parameters
... h = 1e-5
>>> nodes = 2**7
>>> xmin = 0
>>> xmax = 1
>>> a = (xmax - xmin) / (nodes-1)
>>> x = np.linspace(xmin, xmax, nodes)
>>> middle = int(nodes / 2)
>>> timesteps = 5000
...
>>> # Make initial wave function using x-lexicographic ordering of gridpoints
... psi_sq = np.zeros(shape=(nodes, nodes), dtype=np.cfloat)
>>> width = 1
>>> for xx in range(nodes):
...     for yy in range(nodes):
...         psi_sq[xx, yy] = np.exp(-(((xx-middle)**2 + (yy-middle)**2) / (2*width**2)))
...
>>> psi_sq = psi_sq / np.linalg.norm(psi_sq)
>>> psi_sq_plot = np.zeros((nodes, nodes, timesteps+1), dtype=np.cfloat)
>>> psi_sq_plot[:, :, 0] = psi_sq
>>> psi_sq = psi_sq.reshape(-1)  # Make into 1d vector
...
>>> # Make Potential barrier
... V_sq = np.ones((nodes, nodes))
>>> V0 = 1e7
>>> aa = 2**7-2 #width of rectengular potential
>>> bb = 2**7-2 #height of rectengular potential
...
>>> V_sq[middle - 0.5*aa:middle + 0.5*aa, middle - 0.5*bb:middle + 0.5*bb] = 0
>>> V_sq = V0 * V_sq
...
...
>>> # Make left hand side vector by expanding the 1D case (using zero-valued Dirichlet boundary conditions)
... A1D = sp.diags([1 / (4*a**2), 1j/h - 1 / (2*a**2), 1 / (4*a**2)], [-1, 0 ,1], shape=(nodes, nodes))
>>> I = sp.identity(nodes)
>>> A2D = sp.kron(A1D, I) + sp.kron(I, A1D)
>>> A2D -= 0.5 * sp.diags(V_sq.reshape(-1), 0)
...
>>> # Make right hand side vector in the same way
... B1D = sp.diags([-1 / (4*a**2), 1j/h + 1 / (2*a**2), -1 / (4*a**2)], [-1, 0, 1], shape=(nodes, nodes))
>>> B2D = sp.kron(B1D, I) + sp.kron(I, B1D)
>>> B2D += 0.5 * sp.diags(V_sq.reshape(-1), 0)
...
>>> # Using a bicgstab algorithm, solve A*psi(n+1) = B*psi(n) for psi(n+1)
...
... anim_constant = 10
>>> anim_count = 0
>>> animation_frames = int((timesteps+1)/anim_constant)
...
>>> psi_sq_old = psi_sq
>>> psi_sq_new = psi_sq
>>> psi_animate = np.zeros((nodes, nodes, animation_frames), dtype=np.cfloat)
...
>>> for t in range(timesteps):
...     b = B2D.dot(psi_sq_old)
...     psi_sq_new = linalg.bicgstab(A2D, b)[0]
...     psi_sq_new = psi_sq_new / np.linalg.norm(psi_sq_new)
...     psi_sq_old = psi_sq_new
...     psi_sq_plot[:, :, t+1] = psi_sq_new.reshape(nodes, -1)  # reshape for plotting
...     if (t+1) % anim_constant == 0:
...         psi_animate[..., anim_count] = psi_sq_plot[..., t+1]
...         anim_count += 1
```

```python
>>> fig, ax = plt.subplots()
...
>>> xx = range(nodes)
>>> X,Y = np.meshgrid(xx, xx)
>>> Z = np.zeros(X.shape)
>>> z = np.absolute(psi_animate)
>>> V = V_sq
...
>>> anim_sq_well_2d = animation_2d()
>>> plt.show()
```

```python
>>> save_anim(anim_sq_well_2d, 'Square_well_2d')
```

# Calculation of the transmission coefficient

```python
>>> # Set parameters
... h = 1e-5  # timestep size
>>> xmin = 3
>>> xmax = 7
>>> nodes = 10000
...
>>> a = (xmax - xmin) / (nodes-1)  # space between nodes
>>> x = np.linspace(xmin, xmax, nodes)
>>> timesteps = 500
...
>>> # initial wave function
... psi = np.zeros(shape = (nodes, timesteps+1), dtype= np.cfloat)
>>> width = 0.1
>>> p0 = 300
>>> psi[:, 0] = np.exp(-((x-4.5)**2 / (2*width**2)))*np.exp(1j*x*p0)
>>> psi[:, 0] = psi[:, 0] / np.linalg.norm(psi[:, 0])
...
>>> # Set up potential wall with the height of E0 divided by a factor
... E0 = p0**2 / 2
>>> parameters = np.linspace(0.1,4.1,101)
>>> T = np.zeros((len(parameters), 1))
...
>>> for m in range(len(parameters)):
...     V = np.zeros(nodes)
...     V0 = E0 / parameters[m]
...     for xx in range(nodes):
...         V[xx] =  - V0 * (Heaviside(int(nodes/2) - xx) - Heaviside(int(nodes/2) + np.int(7/(a*np.sqrt(2*V0))) - xx))
...
...     # Create matrix A
...     A = sp.diags([1 / (4*a**2), 1j/h - 1/(2*a**2), 1 / (4*a**2)], [-1, 0 ,1], shape=(nodes, nodes)).tolil()
...     A -= 0.5 * sp.diags(V, 0)
...     A = A.tocsc()
...
...     # Create matrix B
...     B = sp.diags([-1 / (4*a**2), 1j/h + 1/(2*a**2), -1 / (4*a**2)], [-1, 0, 1], shape=(nodes, nodes)).tolil()
...     B += 0.5*sp.diags(V, 0)
...     B = B.tocsc()
...
...     anim_constant = 1
...     animation_frames = int((timesteps+1)/anim_constant)
...     anim_count = 0
...
...     for t in range(timesteps):
...         b = B.dot(psi[:, t])
...         psi[:, t+1] = linalg.bicgstab(A, b)[0]
...         psi[:, t+1] = psi[:, t+1] / np.linalg.norm(psi[:, t+1])
...         anim_count += 1
...
...     #calculate the ratio between the transmitted wave and original wave (psi was normalized)
...     T[m] = np.linalg.norm(psi[int(0.5*nodes):, timesteps])**2
```

```python
>>> #Plot the transmission coefficient
... fig = plt.figure(figsize=(6, 3.7))
>>> plt.plot(parameters, T, 'or-', markersize = 3, label = r'$T$')
>>> plt.xlabel(r'$ E / V_0$', fontsize = 9)
>>> plt.ylabel(r'Transmission Coefficient')
>>> plt.show()
```

```python
>>> fig.savefig('Transmission_coefficient.pdf', bbox_inches='tight', pad_inches=0.1)
```

```python
>>> #animate wave through a potential barrier
... psi_animate = psi
>>> fig = plt.figure()
>>> ax = plt.axes(xlim=(3, 7), ylim=(0, 1))
>>> line, = ax.plot([], [], lw=2)
>>> V_x, = ax.plot([],[], lw=2)
...
>>> animtunnel = animation_1D()
>>> plt.show()
```

```python
>>> save_anim(animtunnel,'tunneling_closeup_500t_300p')
```
