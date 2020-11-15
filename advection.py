"""
Advection equation
1-dimensional advection solver for initial conditions f(x,t=0) = x and fixed boundary conditions.
Solves advection equation using FTCS and Lax-Friedrichs methods.

@author: Tristan Ménard
November 12, 2020
"""

import numpy as np
import matplotlib.pyplot as plt

# Set up variables
Ngrid = 50 # number of cells
Nstep = 1000 # number of time steps
dx = 1.0 # grid spacing
dt = 1.0 # size of time step
u = -0.1 # fluid bulk velocity

alpha = 0.5*u*dt/dx

# Verify that the Courant condition is respected: dt < dx/abs(u)
if dt < dx/abs(u):
    print('Courant condition is satisfied. Continuing.')
else:
    print('Warning, Courant condition is not satisfied. Numerical results may be unstable.')

# Initialize f at time t=0, where f(x) = x
x = np.arange(0, Ngrid, dx)
f_FTCS = np.copy(x) # initial state for the FTCS solution
f_LF = np.copy(x) # initial state for the Lax-Friedrichs solution

# Setting up the plot
plt.ion()
fig, ax = plt.subplots(1,2, figsize=(10,5))
# First subplot show the FTCS solution
ax[0].set_title('FTCS method')
ax[0].plot(x, x, label='Initial state')
x1, = ax[0].plot(x, f_FTCS, 'o', label=r'$f(x,t)$')
timestamp = ax[0].text(0.80, 0.02, 't=0.0', transform=ax[0].transAxes)
ax[0].set_xlabel(r'$x$')
ax[0].set_ylabel(r'$f$')
ax[0].set_xlim(0, Ngrid)
ax[0].set_ylim(-Ngrid, 2*Ngrid)
ax[0].legend()
# Second subplot shows the Lax-Friedrichs solution
ax[1].set_title('Lax-Friedrichs method')
ax[1].plot(x, x, label='Initial state')
x2, = ax[1].plot(x, f_LF, 'o', label=r'$f(x,t)$')
ax[1].set_xlabel(r'$x$')
ax[1].set_ylabel(r'$f$')
ax[1].set_xlim(0, Ngrid)
ax[1].set_ylim(-Ngrid, 2*Ngrid)
ax[1].legend()

fig.canvas.draw()

for i in range(Nstep):
    # Implement a step of the FTCS method
    f_FTCS[1:-1] = f_FTCS[1:-1] - alpha*(f_FTCS[2:] - f_FTCS[:-2])
    # Implement a step of the Lax-Friedrichs method
    f_LF[1:-1] = 0.5*(f_LF[2:] + f_LF[:-2]) - alpha*(f_LF[2:] - f_LF[:-2])
    # Update the plot
    x1.set_ydata(f_FTCS)
    x2.set_ydata(f_LF)
    timestamp.set_text(f't={dt*(i+1)}')
    fig.canvas.draw()
    plt.pause(0.001)
