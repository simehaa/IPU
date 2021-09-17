#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <time.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

int main (int argc, char** argv) {
	// Initialization of variables
	int i, j, t, opt;
	int height = 7000;
	int width = 7000;
	int num_iterations = 1000;
	float west, north, east, south;
	float delta = 5.0e-5;
	float epsilon = 0.01;
	float my1 = 0.07;
	float my2 = 0.3;
	float k = 8.0;
	float b = 0.1;
	float a = 0.1;
	float dt = 0.0001;
	float dx = 0.000143;
	float d_dx2 = delta/(dx*dx);
	#ifndef _OPENMP
		clock_t before, after;
	#else
		double before, after;
	#endif
		double time_used;

	// Parsing command-line options
	while ((opt = getopt(argc, argv, "h:w:t:")) != -1) {
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
			default:
				fprintf(stderr, "Usage: %s [-h height] [-w width] [-t no. iterations]\n", argv[0]);
				exit(EXIT_FAILURE);
		}
	}

	// Pad the 2D grid to handle boundary condition
	height += 2;
	width += 2;

	// Allocate matrices
	float **tmp; // temporary pointer to perform pointer swaps
	float **e = (float**) malloc(height*sizeof(float*));
	float **r = (float**) malloc(height*sizeof(float*));
	float **e_bar = (float**) malloc(height*sizeof(float*));
	for (i = 0; i < height; ++i) {
		e[i] = (float*) malloc(width*sizeof(float));
		r[i] = (float*) malloc(width*sizeof(float));
		e_bar[i] = (float*) malloc(width*sizeof(float));
	}

	// Instantiate random values in matrices
	#pragma omp parallel for private(j)
	for (i = 0; i < height; ++i) {
		for (j = 0; j < width; ++j) {
			e[i][j] = (j < width/2) ? 0.0 : 1.0; // left half=0, right half=1
			r[i][j] = (i < height/2) ? 1.0 : 0.0; // top half=1, bottom half=0
			e_bar[i][j] = e[i][j];
		}
	}

	// Start timer
	#ifndef _OPENMP
		before = clock();
	#else
		before = omp_get_wtime();
	#endif

	// Perform computations
	#pragma omp parallel private(i,j,t)
	{
		#ifdef _OPENMP
		#pragma omp single
		{
			printf("Using %d OpenMP threads to parallelize\n", omp_get_num_threads());
			fflush(NULL);
		}
		#endif

		// Perform Forward-Euler Aliev-Panfilov model
		for (t = 0; t < num_iterations; ++t) {

			// Copy immediate inner values to boundaries
			#pragma omp for
			for (i = 1; i < height - 1; ++i) {
				e[i][0] = e[i][2];
				e[i][width - 1] = e[i][width - 3];
			}	

			#pragma omp for
			for (j = 1; j < width - 1; ++j) {
				e[0][j] = e[2][j];
				e[height - 1][j] = e[height - 3][j];
			}

			#pragma omp for
			for (i = 1; i < height - 1; ++i) {
				for (j = 1; j < width - 1; ++j) {

					// Computation of new e
					e_bar[i][j] = e[i][j] + dt*(
					d_dx2*(-4*e[i][j] + west + east + south + north) - 
					k*e[i][j]*(e[i][j] - a)*(e[i][j] - 1) - e[i][j]*r[i][j]
					);

					// Computation of new r
					r[i][j] = r[i][j] + dt*(-epsilon - my1*r[i][j]/(my2 + e[i][j]))*(r[i][j] + k*e[i][j]*(e[i][j] - b - 1));
				}
			}
			#pragma omp single
			{
				// Pointer swap
				tmp = e_bar;
				e_bar = e;
				e = tmp;
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

	// Deallocate matrices
	for (i = 0; i < height; ++i) {
		free(e[i]);
		free(e_bar[i]);
		free(r[i]);
	}
	free(e);
	free(e_bar);
	free(r);

	// Revert to original height and width (which also correpsponds to the actual 
	// height and width that computations were performed on)
	height -= 2; 
	width -= 2;

	// Report parameters and results
	float base = 1e-9*(float)num_iterations/time_used;
	float gflops = base*(float)height*(float)width*28.0;
	float bandwidth = base*sizeof(float)*(float)(height)*(float)(width)*4.0;
	printf("2D Grid           : %d x %d\n", height, width);
	printf("Iterations        : %d\n", num_iterations);
	printf("Time              : %f s\n", time_used);
	printf("Throughput        : %f GFLOPS\n", gflops);
	printf("Minimal Bandwidth : %f GB/s\n", bandwidth);

	return EXIT_SUCCESS;
}