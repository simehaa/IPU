# General Scientific Computing Workloads on the IPU

Computational workloads from general scientific computing have been solved on the [intelligence processing unit](https://www.graphcore.ai/products/ipu) (IPU). This work is part of a master's thesis with University of Oslo (UiO) and Simula, submitted October 2021.

## Contents

* The **Aliev-Panfilov model** is a set of PDEs which model propagation of electric potentials in cardiac tissues. A numerical algorithm, derived by using the forward-Euler solution, was implemented on the IPU and applied on a 2D mesh.
* The **2D heat equation** is a PDE which describes propagation of heat. It was discretized by finite differences, more specifically the explicit scheme found by employing the *forward difference in time* and the *central difference in space*. This became a 5-point stencil-based algorithm, which was applied on a 2D mesh on the IPU.
* The **3D heat equation** extended the heat eq. to 3D meshes. It was implemented as a 7-point stencil-based algorithm for the IPU.
* The [STREAM Triad](http://www.cs.virginia.edu/stream/) benchmark is a common benchmark for CPUs, which solves the kernel `a[i] = b[i] + q*c[i]`. It was implemented as a guide to getting started with IPU programming and served as a benchmark for the peak performance of the IPU.

The 2D and 3D heat equation codes were specifically implemented to be used on both single-IPU and multi-IPU executions.

## Abstract

*The emergence of specialized processors in recent years has largely been driven by providing high computational performance for artificial intelligence (AI) workloads. However, these technological advancements are also of interest for high performance computing (HPC) for general scientific applications. In this thesis, selected stencil-based numerical schemes for solving PDEs have been implemented for execution on the Graphcore IPU, a processor with 1,472 cores and distributed memory chunks of 624 kB located near each core.*

*The heat equation was solved on the IPU for structured 2D, and 3D meshes. The computations and problem sizes were scaled to execute in parallel on up to 16 IPUs. For all executions, the problem size pushed the limit of the available on-chip memory. Additionally, the PDE system of the Aliev-Panfilov model for cardiac excitation was solved for a structured 2D mesh, to demonstrate the IPU's applicability of a real-life physics-based application. The computations involved iteratively applying 5-point and 7-point stencils, for the 2D and 3D systems, respectively. The IPU was demonstrated to achieve remarkable performances, achieving a throughput of up to 1.44 TFLOPS. Careful programming led to an effective use of the distributed in-processor memory of the IPU, which is designed to provide high memory bandwidth. The 3D heat equation reached 5.15 TB/s memory bandwidth on one IPU.*

*The extension to multi-IPU computations also showed performances consistent with the scalable design of IPU systems, so-called IPU-PODs. The scaling of the 3D problem showed a slight decrease in performance per core, whereas the scaling of the 2D problem maintained its performance per core on the multi-IPU executions. Therefore, the multi-IPU performance of the 2D heat equation achieved a better speedup than its 3D counterpart, while both showed performance increases that scaled well. This thesis demonstrates an attempt to apply a specialized AI-processor to selected general scientific computing workloads. In this context, the advantages, challenges, and weaknesses of employing the IPU have been discussed*

## Results

### Hardware and Software Versions
The algorithms were implemented and executed on two processors:
| Processor                      | Language   | Software framework | Cores | Threads | Compiler            |
| ------------------------------ |:---------- | ------------------ | ----- | ------- | ------------------- |
| Two AMD Epyc 7601 32-core CPUs | Standard C | OpenMP 4.5         | 64    | 128     | GCC 11.1.0 with -O3 |
| One Colossus GC200 MK2 IPU     | C++        | Poplar SDK 2.2.0   | 1472  | 8832    | GCC 7.5.0 with -O3  |

**Note**: all executions used single precision 32-bit floats.

### Single-IPU Executions

The heat equation and Aliev-Panfilov model were executed on one IPU and on a Linux server with 2 CPUs. All executions ran for 1000 time steps. The measured performance is shown in the table below:

| Processor | Problem     | Mesh        | Time   | Throughput  | Throughput/core  | Minimal Bandwidth |
| --------- | ----------- | ----------- | ------ | ----------- | ---------------- | ----------------- |
| CPU       | 2D heat eq. | 8000x8000   | 4.05 s | 94.7 GFLOPS | 1.48 GFLOPS | 126.4 GB/s        |
| CPU       | 3D heat eq. | 360x360x360 | 8.83 s | 41.6 GFLOPS | 0.65 GFLOPS | 42.3 GB/s         |
| CPU       | A-P model   | 7000x7000   | 20.5 s | 66.9 GFLOPS | 1.05 GFLOPS | 19.1 GB/s         |
| IPU       | 2D heat eq. | 8000x8000   | 0.30 s | 1.28 TFLOPS | 0.87 TFLOPS | 4.28 TB/s         |
| IPU       | 3D heat eq. | 360x360x360 | 0.26 s | 1.44 TFLOPS | 0.98 TFLOPS | 5.15 TB/s         |
| IPU       | A-P model   | 7000x7000   | 1.09 s | 1.26 TFLOPS | 0.86 TFLOPS | 1.45 TB/s         |

### Multi-IPU Executions

The heat equation was applied to 2D, and 3D meshes on executions ranging from 1 to 16 IPUs. All executions ran for 1000 time steps. The measured performance is shown in the table below:

| No. IPUs | Problem | Mesh        | Time   | Throughput  | Throughput/core  | Minimal Bandwidth |
| -------- | ------- | ----------- | ------ | ----------- | ---------------- | ----------------- |
| 1        | 2D      | 8000x8000   | 0.30 s | 1.28 TFLOPS | 0.87 TFLOPS | 4.28 TB/s         |
| 2        | 2D      | 10000x10000 | 0.28 s | 2.17 TFLOPS | 0.74 TFLOPS | 7.20 TB/s         |
| 4        | 2D      | 14000x14000 | 0.27 s | 4.35 TFLOPS | 0.74 TFLOPS | 14.6 TB/s         |
| 8        | 2D      | 19000x19000 | 0.25 s | 8.59 TFLOPS | 0.74 TFLOPS | 29.1 TB/s         |
| 16       | 2D      | 27000x27000 | 0.25 s | 17.2 TFLOPS | 0.73 TFLOPS | 58.9 TB/s         |
| 1        | 3D      | 360x360x360 | 0.26 s | 1.44 TFLOPS | 0.98 TFLOPS | 5.15 TB/s         |
| 2        | 3D      | 403x403x403 | 0.23 s | 2.30 TFLOPS | 0.78 TFLOPS | 8.30 TB/s         |
| 4        | 3D      | 508x508x508 | 0.26 s | 4.05 TFLOPS | 0.69 TFLOPS | 14.7 TB/s         |
| 8        | 3D      | 640x640x640 | 0.26 s | 7.99 TFLOPS | 0.68 TFLOPS | 29.1 TB/s         |
| 16       | 3D      | 806x806x806 | 0.32 s | 13.2 TFLOPS | 0.56 TFLOPS | 48.0 TB/s         |
