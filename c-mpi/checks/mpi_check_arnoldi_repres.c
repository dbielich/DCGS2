#include "../c-mpi.h"

int mpi_check_arnoldi_repres( MPI_Comm mpi_comm, double *norm_repres, long int m, int mloc, int k, double *b, double *Q, int ldq, double beta, double *H, int ldh ){

	double *bAQ, *d_R;
	double normA, *work;
	int ii, jj;

	bAQ  = (double *) malloc(  mloc * (k+1) * sizeof(double));
	d_R  = (double *) malloc( (k+1) * (k+1) * sizeof(double));

// 	Construct the Krylov subspace
	cblas_dcopy( mloc, b, 1, bAQ, 1);
	for( jj = 0 ; jj < k ; jj++ ) mpi_matvec( MPI_COMM_WORLD, mloc, Q+jj*ldq, bAQ+(jj+1)*mloc );

	// Scratching d_R with beta and H
	d_R[0] = beta;
	for(jj=0;jj<k;jj++) for(ii=0;ii<k+1;ii++) d_R[ ii + (jj+1)*(k+1) ] = H[ ii+jj*ldh ];

	normA = LAPACKE_dlange_work( LAPACK_COL_MAJOR, 'F', mloc, k+1, bAQ, mloc, NULL );
	normA = normA * normA;
	MPI_Allreduce( MPI_IN_PLACE, &normA, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	normA = sqrt(normA);

	work  = (double *) malloc( mloc * ( k+1 ) * sizeof(double));
	LAPACKE_dlacpy_work( LAPACK_COL_MAJOR, 'A', mloc, k+1, Q, ldq, work, mloc ); 
	cblas_dtrmm( CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, mloc, k+1, (+1.0e+00), d_R, k+1, work, mloc );
 	for(jj = 0; jj < k+1; jj++) for(ii = 0; ii < mloc; ii++) work[ ii+jj*mloc ] -= bAQ[ ii+jj*mloc ];
	(*norm_repres) = LAPACKE_dlange_work( LAPACK_COL_MAJOR, 'F', mloc, k+1, work, mloc, NULL );
	free( work );

	(*norm_repres) = (*norm_repres) * (*norm_repres);
	MPI_Allreduce( MPI_IN_PLACE, norm_repres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	(*norm_repres) = sqrt((*norm_repres));

	(*norm_repres )= (*norm_repres) / normA;

	free( bAQ );
	free( d_R );

	return 0;
}

