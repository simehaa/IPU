# 3D Heat Equation

The heat equation was discretized by the finite difference method. The method used an explicit scheme by taking the forward difference in time and central difference in space. This became a 7-point stencil-based numerical algorithm, which was used to solve the heat equation for structured 3D meshes.

## How to Run

```
$ make
$ ./main
```

Additionally, command line arguments can be provided:
```
$ ./main --height 400 --width 400 --depth 400 --num-ipus 2 --num-iterations 1000 --alpha 0.1 --cpu --vertex HeatEquationOptimized
```
* `--height`, `--width`, and `--depth` corresponds to the mesh to be used.
* `--num-ipus` specifies how many IPUs to perform the computations on.
* `--num-iterations` is the number of time steps to perform the computations for.
* `--alpha` is a constant in the heat equation (kappa*dt/h^2).
* `--cpu` will enable an additional CPU execution, and the IPU results will be checked against the CPU results.
* `--vertex` specifies the name of the vertex to use (see *codelets.cpp*). 
