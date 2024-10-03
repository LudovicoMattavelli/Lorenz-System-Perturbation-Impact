# The Lorenz system
The Lorenz system is a set of differential ordinary equations that describe the behave
of a forced dissipative hydrodynamic flow. 
The equations are characterized by three parameters, (σ, b, r), and taking the right values, it results in chaotic solution for the system.
It is important to study the Lorentz system since it is a good example of chaos theory,
characterized by the extreme sensitivity to initial conditions.
## Aim and idea of the code
The following code solves the problem of the integration of the Lorenz system to show the sensitivity of the system to initial conditions and how considering the ensemble can significally reduce the uncertainty on the state of the system. The numerical integration is done using the Euler forward method. 
The study of sensitivity on initial conditions is done by perturbing the initial condition on the x variable first with a 4 predetermined perturbations. Then the system is integrated numerically and at each time the code saves the state of the x-variable and the mean square error in respect to the unperturbed state.
For the ensemble the process is the same, but with 100 random perturbations, uniformly distributed between -
0.75 and 0.75. The time of integration is smaller for the ensemble to reduce the computation power necessary.
