# General Scientific Computing Workloads for the IPU

Computational workloads from general scientific computing have been solved on the [intelligence processing unit](https://www.graphcore.ai/products/ipu) (IPU). This work is part of a master's thesis with University of Oslo (UiO) and Simula, submitted October 2021.

## Contents

* The **Aliev-Panfilov model** is a set of PDEs which model propagation of electric potentials in cardiac tissues. A numerical algorithm, derived by using the forward-Euler solution, was implemented on the IPU and applied on a 2D mesh.
* The **2D heat equation** is a PDE which describes propagation of heat. It was discretized by finite differences, more specifically the explicit scheme found by employing the *forward difference in time* and *central different in space*. This became a 5-point stencil-based algorithm, which was applied on a 2D mesh on the IPU.
* The **3D heat equation** extended the heat eq. to 3D meshes. It was implemented as a 7-point stencil-based algorithm for the IPU.
* The [STREAM Triad](http://www.cs.virginia.edu/stream/) benchmark is a common benchmark for CPUs, which solves the kernel `a[i] = b[i] + q*c[i]`. It was implemented as a guide to getting started with IPU programming and served as a benchmark for the peak performance of the IPU.

The 2D and 3D heat equation codes were specifically implemented to be used on both single-IPU and multi-IPU executions.

## Abstract

*The emergence of specialized processors in recent years has largely been driven by providing high computational performance for artificial intelligence (AI) workloads. However, these technological advancements are also of interest for high performance computing (HPC) for general scientific applications. In this thesis, selected stencil-based numerical schemes for solving PDEs have been implemented for execution on the Graphcore IPU, a processor with 1,472 cores and distributed memory chunks of 624 kB located near each core.*

*The heat equation was solved on the IPU for structured 2D, and 3D meshes. The computations and problem sizes were scaled to execute in parallel on up to 16 IPUs. For all executions, the problem size pushed the limit of the available on-chip memory. Additionally, the PDE system of the Aliev-Panfilov model for cardiac excitation was solved for a structured 2D mesh, to demonstrate the IPU's applicability of a real-life physics-based application. The computations involved iteratively applying 5-point and 7-point stencils, for the 2D and 3D systems, respectively. The IPU was demonstrated to achieve remarkable performances, achieving a throughput of up to 1.44 TFLOPS. Careful programming led to an effective use of the distributed in-processor memory of the IPU, which is designed to provide high memory bandwidth. The 3D heat equation reached 5.15 TB/s memory bandwidth on one IPU.*

*The extension to multi-IPU computations also showed performances consistent with the scalable design of IPU systems, so-called IPU-PODs. The scaling of the 3D problem showed a slight decrease in performance per core, whereas the scaling of the 2D problem maintained its performance per core on the multi-IPU executions. Therefore, the multi-IPU performance of the 2D heat equation achieved a better speedup than its 3D counterpart, while both showed performance increases that scaled well. This thesis demonstrates an attempt to apply a specialized AI-processor to selected general scientific computing workloads. In this context, the advantages, challenges, and weaknesses of employing the IPU have been discussed*
