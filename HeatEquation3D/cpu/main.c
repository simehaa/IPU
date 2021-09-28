#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <time.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

int main (int argc, char** argv)
{
	// Initialization of variables
	int i, j, k, t, height = 360, width = 360, depth = 360, num_iterations = 1000, opt;
	float alpha = 0.1;
	#ifndef _OPENMP
		clock_t before, after;
	#else
		double before, after;
	#endif
		double time_used;

	// Parsing command-line options
	while ((opt = getopt(argc, argv, "h:w:d:t:a:")) != -1) {
		switch (opt) {
			case 'h':
				height = atoi(optarg);
				break;
			case 'w':
				width = atoi(optarg);
				break;
			case 'd':
				depth = atoi(optarg);
				break;
			case 't':
				num_iterations = atoi(optarg);
				break;
			case 'a':
				alpha = atof(optarg);
				break;
			default:
				fprintf(stderr, "Usage: %s [-h height] [-w width] [-d depth] [-t no. iterations] [-a alpha value for heat eq.]\n", argv[0]);
				exit(EXIT_FAILURE);
		}
	}

	// beta reduces the stencil operation to only require 6 flops (instead of 7)
	float beta = (1 - 6*alpha);

	// Allocate matrices
	float ***tmp; // temporary pointer to perform pointer swaps
	float ***a = (float***) malloc(height*sizeof(float**));
	float ***b = (float***) malloc(height*sizeof(float**));
	for (i = 0; i < height; ++i) {
		a[i] = (float**) malloc(width*sizeof(float*));
		b[i] = (float**) malloc(width*sizeof(float*));
		for (j = 0; j < width; ++j) {
			a[i][j] = (float*) malloc(depth*sizeof(float));
			b[i][j] = (float*) malloc(depth*sizeof(float));
		}
	}

	// Instantiate random values in matrices
	#pragma omp parallel for private(j)
	for (i = 0; i < height; ++i) {
		for (j = 0; j < width; ++j) {
			for (k = 0; k < depth; ++k) {
				a[i][j][k] = (float) rand() / (float) (RAND_MAX);
				b[i][j][k] = a[i][j][k];
			}
		}
	}

	// Start timer
	#ifndef _OPENMP
		before = clock();
	#else
		before = omp_get_wtime();
	#endif

	// Perform computations
	#pragma omp parallel private(t,i,j,k)
	{
		#ifdef _OPENMP
		#pragma omp single
		{
			printf("Using %d OpenMP threads to parallelize heat equation\n", omp_get_num_threads());
			fflush(NULL);
		}
		#endif

		// Perform heat equation
		for (t = 0; t < num_iterations; ++t) {
			#pragma omp for
			for (i = 1; i < height - 1; ++i)
				for (j = 1; j < width - 1; ++j)
					for (k = 1; k < depth - 1; ++k)
						b[i][j][k] = beta*a[i][j][k] + alpha*(
							a[i+1][j][k] + a[i-1][j][k] + a[i][j+1][k] + a[i][j-1][k] + a[i][j][k+1] + a[i][j][k-1]);
			#pragma omp single
			{
				// pointer swap
				tmp = b;
				b = a;
				a = tmp;
			}
		}
	}

	// End timer and evaluate time used
	#ifndef _OPENMP
		after = clock();
		time_used = (float) (after - before) / (float) CLOCKS_PER_SEC;
		#else
		after = omp_get_wtime();
		time_used = after - before;
	#endif

	// deallocate matrices
	for (i = 0; i < height; ++i) {
		for (j = 0; j < width; ++j) {
			free(a[i][j]);
			free(b[i][j]);
		}
		free(a[i]);
		free(b[i]);
	}
	free(a);
	free(b);

	// Report parameters and results
	float base = 1e-9*(float)num_iterations/time_used;
	float gflops = base*(float)(height-2)*(float)(width-2)*(float)(depth-2)*8.0;
	float bandwidth = base*sizeof(float)*(float)height*(float)width*(float)depth*2.0;
	printf("3D Grid           : %d x %d x %d\n", height, width, depth);
	printf("Iterations        : %d\n", num_iterations);
	printf("alpha             : %g\n", alpha);
	printf("Time              : %f s\n", time_used);
	printf("Throughput        : %f GFLOPS\n", gflops);
	printf("Minimal Bandwidth : %f GB/s\n", bandwidth);

	return EXIT_SUCCESS;
}