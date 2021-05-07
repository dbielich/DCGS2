#include "../c-mpi.h"

int mpi_check_manteuffel_arnoldi_repres( MPI_Comm mpi_comm, double *norm_repres, long int m, int mloc, int k, double h, double BEETA, double *b, double *Q, int ldq, double beta, double *H, int ldh ){

	double *bAQ, *d_R;
	double normA, *work;
	double *y, *v;
	int ii, jj, n;

	n = sqrt( m );
	double *x_notlocal;
	x_notlocal = (double *) malloc( 2 * n * sizeof(double));

	bAQ = (double *) malloc(  mloc * (k+1) * sizeof(double));
	d_R = (double *) malloc( (k+1) * (k+1) * sizeof(double));
	y = (double *) malloc(  mloc * sizeof(double));
	v = (double *) malloc(  mloc * sizeof(double));

// 	Construct the Krylov subspace
	cblas_dcopy( mloc, b, 1, bAQ, 1);
	for( jj = 0 ; jj < k ; jj++ ){
		mpi_matvec_manteuffel_M( MPI_COMM_WORLD, m, Q+jj*ldq, y );
		mpi_matvec_manteuffel_M_nblock_eo( MPI_COMM_WORLD, m, Q+jj*ldq, y, x_notlocal );
		mpi_matvec_manteuffel_N( MPI_COMM_WORLD, m, Q+jj*ldq, v );
		mpi_matvec_manteuffel_N_nblock_eo( MPI_COMM_WORLD, m, Q+jj*ldq, v, x_notlocal );

		for( ii=0;ii<mloc;ii++ ){ bAQ[ii+(jj+1)*mloc] = ( 1 / ( h * h ) ) * y[ii] + ( BEETA / ( 2 * h ) ) * v[ii]; }
	}

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

	free( y );
	free( v );
	free( x_notlocal );
	free( bAQ );
	free( d_R );

	return 0;
}

