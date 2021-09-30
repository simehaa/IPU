# STREAM Triad

A simple axpy (a times x plus y) kernel,
```
for (int i = 0; i < n; ++i) {
    a[i] = b[i] + c[i]*scalar;
}
```
implemented in a highly parallelized fashion on the IPU. The most optimized implementation achieved:
* **7.76 TFLOPS** computational throughput (single precision).
* **46.55 TB/s** memory bandwidth. 
This is roughly 99.1 % of theoretical best performance for the processor.

## How to Run

```
$ make
$ ./main
```
