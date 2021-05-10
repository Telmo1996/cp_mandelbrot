/*The Mandelbrot set is a fractal that is defined as the set of points c
in the complex plane for which the sequence z_{n+1} = z_n^2 + c
with z_0 = 0 does not tend to infinity.*/

/*This code computes an image of the Mandelbrot set.*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

#define DEBUG 1

#define          X_RESN  1024  /* x resolution */
#define          Y_RESN  1024  /* y resolution */

/* Boundaries of the mandelbrot set */
#define           X_MIN  -2.0
#define           X_MAX   2.0
#define           Y_MIN  -2.0
#define           Y_MAX   2.0

/* More iterations -> more detailed image & higher computational cost */
#define   maxIterations  1000

typedef struct complextype
{
	float real, imag;
} Compl;

static inline double get_seconds(struct timeval t_ini, struct timeval t_end)
{
	return (t_end.tv_usec - t_ini.tv_usec) / 1E6 +
         (t_end.tv_sec - t_ini.tv_sec);
}

int main (int argc, char *argv[])
{
	MPI_Init(&argc, &argv);

	/* Mandelbrot variables */
	int i, j, k;
	Compl   z, c;
	float   lengthsq, temp;
	int *vres, *res[Y_RESN];
	int *vresParcial, *resParcial[Y_RESN];
	int numProcs, rank;
	int *filasPorProceso, filaInicio, filaFin;
	int *filasIniciales, *filasFinales;

	/* Timestamp variables */
	struct timeval  ti, tf;
	float			*tiempos;
	float maxt = 0.0;
	float tsend;
	float ttotal = 0.0;

	float maxFlops = 0.0;
	int flops = 0;
	int flopsTotal = 0;
	int *arrFlops;
	float balanceo;


	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int recvcounts[numProcs];
	int displs[numProcs];

	filasIniciales = (int*) malloc(numProcs*sizeof(int));
	filasFinales = (int*) malloc(numProcs*sizeof(int));

	//Calcular filas iniciales y finales de cada proceso
	for(i=0; i<numProcs; i++){
		if(i < Y_RESN%numProcs){
			filasPorProceso[i] = Y_RESN % numProcs +1;
			filasIniciales[i] = i*filasPorProceso[i];
			filasFinales[i] = filasIniciales[i]+filasPorProceso[i]-1;
		}else{
			filasPorProceso[i] = Y_RESN / numProcs;
			filasIniciales[i] = filasFinales[i-1]+1;
			filasFinales[i] = filasIniciales[i]+filasPorProceso[i]-1;
		}
		recvcounts[i]=filasPorProceso[i]*X_RESN;
		displs[i] = filasIniciales[i]*X_RESN;
	}

	if(rank == 0){	
		/* Allocate result matrix of Y_RESN x X_RESN */
		vres = (int *) malloc(Y_RESN * X_RESN * sizeof(int));
		if (!vres){
			fprintf(stderr, "Error allocating memory\n");
			return 1;
		}
		for (i=0; i<Y_RESN; i++)
			res[i] = vres + i*X_RESN;

		tiempos = (float *) malloc(numProcs * sizeof(float));
		arrFlops = (int *) malloc(numProcs * sizeof(int));
	}

	vresParcial = (int *) malloc(X_RESN * (filasPorProceso[rank]) * sizeof(int));
	if (!vresParcial){
		fprintf(stderr, "Error allocating memory\n");
		return 1;
	}
	for (i=0; i<filasPorProceso[rank]; i++)
		resParcial[i] = vresParcial + i*X_RESN;

	/* Start measuring time */
	gettimeofday(&ti, NULL);

	filaInicio = filasIniciales[rank];
	filaFin = filasFinales[rank];

	/* Calculate and draw points */
	for(i=filaInicio; i < filaFin; i++)
	{
		for(j=0; j < X_RESN; j++)
		{
			z.real = z.imag = 0.0;
			c.real = X_MIN + j * (X_MAX - X_MIN)/X_RESN;
			c.imag = Y_MAX - i * (Y_MAX - Y_MIN)/Y_RESN;
			k = 0;

			flops += 8;

			do
			{    /* iterate for pixel color */
				temp = z.real*z.real - z.imag*z.imag + c.real;
				z.imag = 2.0*z.real*z.imag + c.imag;
				z.real = temp;
				lengthsq = z.real*z.real+z.imag*z.imag;
				k++;

				flops += 10;
			} while (lengthsq < 4.0 && k < maxIterations);

			if (k >= maxIterations) resParcial[i%filasPorProceso[rank]][j] = 0;
			else resParcial[i%filasPorProceso[rank]][j] = k;
		}
	}

	/* End measuring time */
	gettimeofday(&tf, NULL);

	tsend = get_seconds(ti,tf);
	MPI_Gather(&tsend, 1, MPI_FLOAT, 
				tiempos, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Gather(&flops, 1, MPI_INT, 
				arrFlops, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gatherv(vresParcial, recvcounts[rank], MPI_INT, 
				vres, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

	if(rank == 0){
		/* Print result out */
		if( DEBUG) {
			for(i=0;i<Y_RESN;i++) {
				for(j=0;j<X_RESN;j++)
					printf("%3d ", res[i][j]);
				printf("\n");
			}
		}


		for(i=0; i<numProcs; i++){
			//Imprimir tiempos
			fprintf (stderr, "(PERF) Time (seconds) = %lf, num Operaciones = %i\n", tiempos[i], arrFlops[i]);

			flopsTotal += arrFlops[i];
			if(arrFlops[i] > maxFlops)
				maxFlops = arrFlops[i];

			if(tiempos[i] > maxt)
				maxt = tiempos[i];

			ttotal += tiempos[i];
		}

		fprintf (stderr, "\nTiempo total = %lf\n", maxt);

		balanceo = (flopsTotal) / ((float)maxFlops * (float)numProcs);
		fprintf (stderr, "Balanceo = %lf\n", balanceo);

	}
	

	free(vres);
	free(tiempos);
	
	MPI_Finalize();

	return 0;
}

