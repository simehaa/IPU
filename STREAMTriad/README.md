# STREAM Triad

A simple axpy kernel,
```
for (int i = 0; i < n; ++i) {
    a[i] = b[i] + c[i]*scalar;
}
```
implemented in a highly parallelized fashion on the ipu.

## How to Run

```
$ make
$ ./main
```

## See the Assembly Code

Generate the assembly code by
```
$ popc --S codelets.cpp -o codelets.S -O3 -target=ipu1
```