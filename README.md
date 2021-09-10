# IPU Codes

## Folder Overview

* AlievPanfilovModel: the Forward-Euler solution of the Aliev-Panfilov model: simulation of electric pulses in cardiac tissues.
* HeatEquation2D: explicit scheme for the 2D heat equation (isotropic diffusion).
* HeatEquation3D: explicit scheme for the 3D heat equation (isotropic diffusion).
* HeatEquationMultiIPU: like HeatEquation3D but specifically implemented to be solved by 1 or multiple IPUs.
* STREAMTriad: the STREAM Triad benchmark for the IPU. Solving `a[i] = b[i] + q*c[i]`, highly parallelized.