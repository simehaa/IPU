# The Aliev-Panfilov Model

The Aliev-Panfilov model is a set of PDEs that model electric pulse propagation in cardiac tissue. It was constructed to include the qualitative behavior of cardiac tissue, while being computationally feasible. The implemented algorithm is the forward-Euler solution to the so-called monodomain model, which accounts for the transmembrane potential.

Numerically, the model is implemented as two 2D meshes, *e* and *r*, which affect each other:
* The *e* mesh is updated by using a 5-point stencil, including 1 point from *r*. 
* The *r* mesh is updated by using 1 point from both *e* and *r*.

## How to Run

```
$ make
$ ./main
```

Additionally, command line arguments can be provided:
```
$ ./main --height 8000 --width 8000 --num-iterations 1000 --cpu 
```
* `--height` and `--width` corresponds to the mesh to be used.
* `--num-iterations` is the number of time steps to perform the computations for.
* `--cpu` will enable an additional CPU execution, and the IPU results will be checked against the CPU results.
* `--my1`, `--my2`, `--k`, `--epsilon`, `--b`, `--a`, `--dt`, `--h`, and `--delta` are constants of the Aliev-Panfilov model. They have reasonable default values.
