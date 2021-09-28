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
	int i, j, k, height = 8000, width = 8000, num_iterations = 1000, opt;
	float alpha = 0.1;
	#ifndef _OPENMP
		clock_t before, after;
	#else
		double before, after;
	#endif
		double time_used;

	// Parsing command-line options
	while ((opt = getopt(argc, argv, "h:w:t:a:")) != -1) {
		switch (opt) {
			case 'h':
				height = atoi(optarg);
				break;
			case 'w':
				width = atoi(optarg);
				break;
			case 't':
				num_iterations = atoi(optarg);
				break;
			case 'a':
				alpha = atof(optarg);
				break;
			default:
				fprintf(stderr, "Usage: %s [-h height] [-w width] [-t no. iterations] [-a alpha value for heat eq.]\n", argv[0]);
				exit(EXIT_FAILURE);
		}
	}

	// beta reduces the stencil operation to only require 6 flops (instead of 7)
	float beta = (1 - 4*alpha);

	// Allocate matrices
	float **tmp; // temporary pointer to perform pointer swaps
	float **a = (float**) malloc(height*sizeof(float*));
	float **b = (float**) malloc(height*sizeof(float*));
	for (i = 0; i < height; ++i) {
		a[i] = (float*) malloc(width*sizeof(float));
		b[i] = (float*) malloc(width*sizeof(float));
	}

	// Instantiate random values in matrices
	#pragma omp parallel for private(j)
	for (i = 0; i < height; ++i) {
		for (j = 0; j < width; ++j) {
			a[i][j] = (float) rand() / (float) (RAND_MAX);
			b[i][j] = a[i][j];
		}
	}

	// Start timer
	#ifndef _OPENMP
		before = clock();
	#else
		before = omp_get_wtime();
	#endif

	// Perform computations
	#pragma omp parallel private(i,j,k)
	{
		#ifdef _OPENMP
		#pragma omp single
		{
			printf("Using %d OpenMP threads to parallelize heat equation\n", omp_get_num_threads());
			fflush(NULL);
		}
		#endif

		// Perform heat equation
		for (k = 0; k < num_iterations; ++k) {
			#pragma omp for
			for (i = 1; i < height - 1; ++i)
				for (j = 1; j < width - 1; ++j)
					b[i][j] = beta*a[i][j] + alpha*(a[i-1][j] + a[i][j-1] + a[i][j+1] + a[i+1][j]);

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
		free(a[i]);
		free(b[i]);
	}
	free(a);
	free(b);

	// Report parameters and results
	printf("2D Grid           : %d x %d\n", height, width);
	printf("Iterations        : %d\n", num_iterations);
	printf("alpha             : %g\n", alpha);
	printf("Time              : %f s\n", time_used);
	printf("Throughput        : %f GFLOPS\n", 1e-9*(float)num_iterations*(float)(height-2)*(float)(width-2)*6/time_used);
	printf("Minimal Bandwidth : %f GB/s\n", 1e-9*sizeof(float)*(float)num_iterations*(float)height*(float)width*2.0/time_used);

	return EXIT_SUCCESS;
}