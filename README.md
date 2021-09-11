# IPU Codes

## Folder Overview

* AlievPanfilovModel: the Forward-Euler solution of the Aliev-Panfilov model, which describes propagation of electric potentials in cardiac tissues. A 5-point stencil algorithm using two 2D meshes.
* HeatEquation2D: explicit scheme for the 2D heat equation (isotropic diffusion). A 5-point stencil algorithm on a 2D mesh.
* HeatEquation3D: explicit scheme for the 3D heat equation (isotropic diffusion). A 7-point stencil algorithm on a 3D mesh.
* STREAMTriad: the STREAM Triad benchmark for the IPU: solving the kernel `a[i] = b[i] + q*c[i]`.