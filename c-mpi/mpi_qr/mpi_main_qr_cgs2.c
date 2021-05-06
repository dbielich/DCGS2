#include "../c-mpi.h"

int main(int argc, char ** argv) {


	double *A, *Q, *R, *work;
	double norm_orth=0.0e+00, norm_repres=0.0e+00;
	double elapse, nrmA;
	int lda, ldq, ldr;
	int m, n, i, j, testing;
	int my_rank, pool_size, seed, mloc;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &pool_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    	m         = 27*pool_size;
	n         = 11;
	testing   =  0;

	for(i = 1; i < argc; i++){
		if( strcmp( *(argv + i), "-testing") == 0) {
			testing = atoi( *(argv + i + 1) );
			i++;
		}
		if( strcmp( *(argv + i), "-m") == 0) {
			m = atoi( *(argv + i + 1) );
			i++;
		}
		if( strcmp( *(argv + i), "-n") == 0) {
			n = atoi( *(argv + i + 1) );
			i++;
		}
	}

	seed = my_rank*m*m; srand(seed);
	mloc = m / pool_size; if ( my_rank < m - pool_size*mloc ) mloc++;

	lda = mloc; ldq = mloc; ldr = n;

	int stop=0; if( mloc < n ){ stop = 1; }
	MPI_Allreduce( MPI_IN_PLACE, &stop, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	if ( stop > 0 ){ printf("\n\n We need n <= mloc: mloc = %3d & n = %3d\n\n",mloc,n); MPI_Finalize(); return 0; }

	work = (double *) malloc( 2*n * sizeof(double));

	Q = (double *) malloc( ldq * n * sizeof(double));
	A = (double *) malloc( lda * n * sizeof(double));
	R = (double *) malloc( n * n * sizeof(double));

 	for(i = 0; i < lda * n; i++)
		*(A + i) = (double)rand() / (double)(RAND_MAX) - 0.5e+00;

	LAPACKE_dlacpy_work( LAPACK_COL_MAJOR, 'A', mloc, n, A, lda, Q, ldq );
	nrmA = LAPACKE_dlange_work( LAPACK_COL_MAJOR, 'F', mloc, n, A, lda, NULL );

	MPI_Barrier( MPI_COMM_WORLD );
	elapse = -MPI_Wtime();

	for( j=0; j<n; j++ ){
	   mpi_orth_cgs2_threesynch_append1( mloc, j, Q, mloc, R+j*ldr, work, MPI_COMM_WORLD);
	}


	MPI_Barrier( MPI_COMM_WORLD );
	elapse += MPI_Wtime();

	if ( testing ){

		if( my_rank == 0 ) printf("Classical Gram Schmidt with reorthogonalization (CGS2) \n");
		if( my_rank == 0 ) printf("-----------------------------------------------------\n");
		mpi_check_qq_orth( MPI_COMM_WORLD, &norm_orth, mloc, n, Q, ldq );
		if( my_rank == 0 ) printf("      || I - Q^T Q ||         = %5.1e \n ",norm_orth);
		mpi_check_qr_repres( MPI_COMM_WORLD, &norm_repres, mloc, n, A, lda, Q, ldq, R, ldr );
		if( my_rank == 0 ) printf("     || A - Q R || / || A || = %5.1e\n",norm_repres);
		if( my_rank == 0 ) printf("-----------------------------------------------------\n");
 		if( my_rank == 0 ) printf("     m = %6d,  n = %6d,  pool size = %3d\n", m, n, pool_size);		
		if( my_rank == 0 ) printf("-----------------------------------------------------\n");
 		if( my_rank == 0 ) printf("          elapsed time = %15.8f\n", elapse);		
		if( my_rank == 0 ) printf("-----------------------------------------------------\n");

	} else {
 		if( my_rank == 0 ) printf("%6d %6d %3d %15.8f\n", m, n, pool_size, elapse);
	}

	free( A );
	free( Q );
	free( R );
	free( work );

        MPI_Finalize();
	return 0;

}
