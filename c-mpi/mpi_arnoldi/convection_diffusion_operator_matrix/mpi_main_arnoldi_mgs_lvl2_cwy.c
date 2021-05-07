#include "../../c-mpi.h"

int main(int argc, char ** argv) {

	double *Q, *T, *H, *b, beta;
	double *y, *v, BEETA, L, h;
	double norm_orth, norm_repres;
	double elapse, tol;
	long int m;
	int n, k, i, j, testing;
	int my_rank, pool_size, seed, mloc;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &pool_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    	n         = 2*pool_size;
	k         = 11;
	testing   =  0;
	tol       = 1.00e-7;
	BEETA      = 0.5;

	for(i = 1; i < argc; i++){
		if( strcmp( *(argv + i), "-testing") == 0) {
			testing = atoi( *(argv + i + 1) );
			i++;
		}
		if( strcmp( *(argv + i), "-n") == 0) {
			n = atoi( *(argv + i + 1) );
			i++;
		}
		if( strcmp( *(argv + i), "-k") == 0) {
			k = atoi( *(argv + i + 1) );
			i++;
		}
		if( strcmp( *(argv + i), "-tol") == 0) {
			tol = atof( *(argv + i + 1) );
			i++;
		}
		if( strcmp( *(argv + i), "-b") == 0) {
			BEETA = atof( *(argv + i + 1) );
			i++;
		}
	}
	m = ( n * n );
        L = n+1;
 	h = L / (n+1);

	seed = my_rank*m*m; srand(seed);
	mloc = m / pool_size; if ( my_rank < m - pool_size*mloc ) mloc++;

	int stop=0; if( mloc < k ){ stop = 1; }
	MPI_Allreduce( MPI_IN_PLACE, &stop, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	if ( stop > 0 ){ printf("\n\n We need k <= mloc\n\n"); MPI_Finalize(); return 0; }
	if ( pool_size >= n ){ printf("\n\n We need pool_size < n\n\n"); MPI_Finalize(); return 0; }

	y = (double *) malloc( mloc * sizeof(double));
	v = (double *) malloc( mloc * sizeof(double));

	b = (double *) malloc( mloc * sizeof(double));
	H = (double *) malloc( (k+1) * ( k ) * sizeof(double));
	Q = (double *) malloc( mloc * (k+1) * sizeof(double));
	T = (double *) malloc( (k+1) * (k+1) * sizeof(double)); 

	double *x_notlocal;
	x_notlocal = (double *) malloc( 2 * n * sizeof(double));	

 	for(i = 0; i < mloc; i++){
		*(b + i) = ( (double) 1 );
	}

	MPI_Barrier( MPI_COMM_WORLD );
	elapse = -MPI_Wtime();

	cblas_dcopy( mloc, b, 1, Q, 1);
	mpi_orth_mgs_lvl2_cwy_threesynch_append1( mloc, 0, Q, mloc, T, k+1, &beta, MPI_COMM_WORLD);

	for( j = 1 ; j < k+1 ; j++ ){

		mpi_matvec_manteuffel_M( MPI_COMM_WORLD, m, Q+(j-1)*mloc, y );
		mpi_matvec_manteuffel_M_nblock_eo( MPI_COMM_WORLD, m, Q+(j-1)*mloc, y, x_notlocal );
		mpi_matvec_manteuffel_N( MPI_COMM_WORLD, m, Q+(j-1)*mloc, v );
		mpi_matvec_manteuffel_N_nblock_eo( MPI_COMM_WORLD, m, Q+(j-1)*mloc, v, x_notlocal );

		for( i=0;i<mloc;i++ ){ Q[i+j*mloc] = ( 1 / ( h * h ) ) * y[i] + ( BEETA / ( 2 * h ) ) * v[i]; }

		mpi_orth_mgs_lvl2_cwy_threesynch_append1( mloc, j, Q, mloc, T, k+1, H+(j-1)*(k+1), MPI_COMM_WORLD);
	}

	MPI_Barrier( MPI_COMM_WORLD );
	elapse += MPI_Wtime();

	if ( testing ){

		mpi_check_qq_orth( MPI_COMM_WORLD, &norm_orth, mloc, k+1, Q, mloc  );
		mpi_check_manteuffel_arnoldi_repres( MPI_COMM_WORLD, &norm_repres, m, mloc, k, h, BEETA, b, Q, mloc, beta, H, k+1 );
		if( my_rank == 0 ){ printf("%3d %8ld %6d %6.2f  %6.2e %6.6e  %8.4e %8.4e", 
					pool_size, m, k, BEETA, tol, elapse, norm_repres, norm_orth );
		}

		if( my_rank == 0 ) printf("\n");


	}
	if ( !testing ){
		if( my_rank == 0 ){ printf("%3d %8ld %6d %6.2f  %6.2e %6.6e", pool_size, m, k, BEETA, tol, elapse ); }
		if( my_rank == 0 ) printf("\n");
	}

	free( y );
	free( v );
	free( x_notlocal );

	free( Q );
	free( T );
	free( H );
	free( b );

        MPI_Finalize();
	return 0;

}
