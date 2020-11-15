"""
1-dimensional hydrodynamics solver
1-dimensional hydrodynamics solver using the donor cell advection scheme.
Initial condition is uniform density gas (in 1D, no gravity) with some small Gaussian perturbation in density.

@author: Tristan Ménard
November 12, 2020
"""

import numpy as np
import matplotlib.pyplot as plt

# Set up variables
Ngrid = 200 # number of cells
Nstep = 1000 # number of time steps
dx = 0.25 # grid spacing
# (Note: time step size will be re-calculated for every iteration of the solver so that the Courant condition is respected.)
courant = 0.5 # Courant number used by this numerical solver to determine the max allowed time step (Courant # < 1 for numerical stability)
u0 = 0.0 # set initial bulk fluid velocity to zero
cs = 1.0 # speed of sound

x = np.arange(0, Ngrid*dx, dx)

# Gas density is initially uniform with a small Gaussian perturbation
rho0 = 0.1 # uniform initial gas density (baseline of the Gaussian function)
a = 1.0 # amplitude of the Gaussian perturbation
mu = Ngrid*dx/2 # central location of the perturbation
sigma = 0.75 # width of the perturbation
rho = a*np.exp(-0.5*(x-mu)**2/sigma**2) + rho0 # gas density with perturbation
f1 = np.copy(rho) # by definition f1 is the density
f2 = np.copy(u0*f1) # by definition of f2

# Setting up the plot
plt.ion()
fig = plt.figure()
ax = plt.gca()
plt.title('1-dimensional hydro solver')
plt.plot(x, rho, label='Initial state')
x1, = plt.plot(x, f1, label=r'$f_1(x,t)$')
timestamp = plt.text(0.04, 0.94, 't=0.0', transform=ax.transAxes)
plt.xlabel(r'$x$')
plt.ylabel(r'Density, $f_1$')
plt.xlim(0, Ngrid*dx)
plt.ylim(np.max([0, rho0-a-a/2]), rho0+a+a/2)
plt.legend()

fig.canvas.draw()

t = 0 # initialize time at t = 0
for i in range(Nstep):
    # Calculate the allowed time step size according to the Courant condition and the chosen Courant number
    dt = courant*np.min(dx/(cs+np.abs(f2/f1)))
    t += dt

    # Compute the velocities at the cell interfaces
    ui = np.zeros(Ngrid+1)
    ui[1:-1] = 0.5*(f2[1:]/f1[1:] + f2[:-1]/f1[:-1])
    idx = ui[1:-1] > 0 # find the indices where the interface velocity is positive

    # Compute the flux J1
    J1 = np.zeros(Ngrid+1)
    J1[1:-1][idx] = ui[1:-1][idx]*f1[:-1][idx] # when the interface velocity is positive
    J1[1:-1][~idx] = ui[1:-1][~idx]*f1[1:][~idx] # when the interface velocity is negative

    # Update the density f1
    f1 = f1 - dt/dx*(J1[1:] - J1[:-1])
    # Apply reflective boundary conditions
    f1[0] = f1[0] - dt/dx*J1[0]
    f1[-1] = f1[-1] + dt/dx*J1[-1]

    # Compute the flux J2
    J2 = np.zeros(Ngrid+1)
    J2[1:-1][idx] = f2[:-1][idx]**2/f1[:-1][idx] # Note: f2/f1 is equivalent to u*f2
    J2[1:-1][~idx] = f2[1:][~idx]**2/f1[1:][~idx]

    # Update the momentum f2
    f2 = f2 - dt/dx*(J2[1:] - J2[:-1])
    # Add the pressure forces from the Euler equation to f2
    f2[1:-1] = f2[1:-1] - 0.5*dt/dx*cs**2*(f1[2:] - f1[:-2])
    # Apply reflective boundary conditions
    f2[0] = f2[0] - 0.5*dt/dx*cs**2*(f1[1] - f1[0])
    f2[-1] = f2[-1] - 0.5*dt/dx*cs**2*(f1[-1] - f1[-2])

    # Update the plot
    x1.set_ydata(f1)
    timestamp.set_text(f't={t:.3f}')
    fig.canvas.draw()
    plt.pause(0.001)
