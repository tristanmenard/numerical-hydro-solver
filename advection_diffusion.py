"""
Advection-diffusion equation
1-dimensional advection-diffusion solver for initial conditions f(x,t=0) = x and fixed boundary conditions.
Solves advection-diffusion equation using the Lax-Friedrichs method and implicit operator splitting.

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

# Choose two different diffuion coefficients
D1 = 0.3
D2 = 3.0

beta1 = D1*dt/dx**2
beta2 = D2*dt/dx**2

# Verify that the Courant condition is respected: dt < dx/abs(u)
if dt < dx/abs(u):
    print('Courant condition is satisfied. Continuing.')
else:
    print('Warning, Courant condition is not satisfied. Numerical results may be unstable.')

# Initialize f at time t=0, where f(x) = x
x = np.arange(0, Ngrid, dx)
f1 = np.copy(x) # initial state with the 1st diffusion coefficient (D1)
f2 = np.copy(x) # initial state with the 2nd diffusion coefficient (D2)

# Initialize A matrix for the implicit method for each diffusion coefficient
A1 = (1+2*beta1)*np.eye(Ngrid) - beta1*np.eye(Ngrid, k=1) - beta1*np.eye(Ngrid, k=-1)
A2 = (1+2*beta2)*np.eye(Ngrid) - beta2*np.eye(Ngrid, k=1) - beta2*np.eye(Ngrid, k=-1)

# Apply fixed (no-slip) boundary conditions on matrices A1 and A2
A1[0][0] = 1
A1[0][1] = 0
A1[-1][-1] = 1
A1[-1][-2] = 0

A2[0][0] = 1
A2[0][1] = 0
A2[-1][-1] = 1
A2[-1][-2] = 0

# Setting up the plot
plt.ion()
fig, ax = plt.subplots(1,2, figsize=(10,5))
# First subplot shows the solution with diffusion coefficient D1
ax[0].set_title(f'$D={D1}$')
ax[0].plot(x, x, label='Initial state')
x1, = ax[0].plot(x, f1, 'o', label=r'$f(x,t)$')
timestamp = ax[0].text(0.04, 0.94, 't=0.0', transform=ax[0].transAxes)
ax[0].set_xlabel(r'$x$')
ax[0].set_ylabel(r'$f$')
ax[0].set_xlim(0, Ngrid)
ax[0].set_ylim(0, Ngrid)
ax[0].legend(loc='lower right')
# Second subplot shows the solution with diffusion coefficient D2
ax[1].set_title(f'$D={D2}$')
ax[1].plot(x, x, label='Initial state')
x2, = ax[1].plot(x, f2, 'o', label=r'$f(x,t)$')
ax[1].set_xlabel(r'$x$')
ax[1].set_ylabel(r'$f$')
ax[1].set_xlim(0, Ngrid)
ax[1].set_ylim(0, Ngrid)
ax[1].legend(loc='lower right')

fig.canvas.draw()

# Implicitly solve the advection-diffusion equation for each time step using operator splitting and the Lax-Friedrichs method
for i in range(Nstep):
    # Update the grid with the diffusion term
    f1 = np.linalg.solve(A1, f1)
    f2 = np.linalg.solve(A2, f2)
    # Calculate and add the advection term
    f1[1:-1] = 0.5*(f1[2:] + f1[:-2]) - alpha*(f1[2:] - f1[:-2])
    f2[1:-1] = 0.5*(f2[2:] + f2[:-2]) - alpha*(f2[2:] - f2[:-2])
    # Update the plot
    x1.set_ydata(f1)
    x2.set_ydata(f2)
    timestamp.set_text(f't={dt*(i+1)}')
    fig.canvas.draw()
    plt.pause(0.001)
