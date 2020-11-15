# PHYS 643: Astrophysical Fluids
# Problem Set 5

Tristan Ménard

November 15, 2020

Python version: 3.8.3

## Files
* advection.py: 1-dimensional advection solver for initial conditions f(x, t=0) = x and fixed boundary conditions. Solves the advection equation using both FTCS and Lax-Friedrichs methods.
* advection_diffusion.py: 1-dimensional advection-diffusion solver for initial conditions f(x, t=0) = x and fixed boundary conditions. Solves the advection-diffusion equation using the Lac-Friedrichs method and implicit operator splitting.
* hydro_solver.py: 1-dimensional hydrodynamics solver using the donor cell advection scheme. Initial condition is uniform density gas with some small Gaussian perturbation along with no initial bulk fluid velocity.

## Problems
### 1. Advection equation
Code: advection.py

### 2. Advection-diffusion equation
Code: advection_diffusion.py

A higher diffusion coefficient means that the fluid departs less from its initial state.

### 3. 1-dimensional hydro solver
Code: hydro_solver.py

When the amplitude (a) of the Gaussian perturbation is smaller than the base density level of the fluid (rho0), there are density waves that travel outwards from the perturbation that are followed by smaller waves or ripples.
When the amplitude (a) of the Gaussian perturbation is greater than the base density level of the fluid (rho0), there is a sharp wavefront, or shock, that travels outwards from the pertubation. The shock is not followed by smaller ripples.
By running the hydro solver with a variety of initial parameter settings, I found that the sound speed (cs) is the parameter that most significantly affects the width of the shock. When the sound speed is large, the shock wave propagates faster and the width of the shock is larger.
To a lesser degree, both the ratio of the perturbation amplitude to the base fluid density (a/rho0) and the initial width of the perturbation (sigma) also affect the width of the shock. In both cases, the width of the shock is larger when these values are large.
