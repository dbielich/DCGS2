#include "../c-mpi.h"

int mpi_orth_hh_lvl2_fivesynch_append1( MPI_Comm mpi_comm, int mloc, int j, double *A, int lda, double *T, int ldt, double *q, double *r ){

	double norma, norma2;

	int i;

	double *buff;
	int my_rank;

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	buff = (double *) malloc( 2 * sizeof(double));

	if( j == 0 ){

		if( my_rank == 0 ) buff[0] = A[0];
		else               buff[0] = 0.0e+00;

		if( my_rank == 0 ) norma2 = cblas_ddot( mloc-1, A+1, 1, A+1, 1 );
		else               norma2 = cblas_ddot( mloc, A, 1, A, 1 ); 

		buff[1] = norma2;

		MPI_Allreduce( MPI_IN_PLACE, buff, 2, MPI_DOUBLE, MPI_SUM, mpi_comm);

		if( my_rank == 0 ) r[0] = A[0];
		else               r[0] = buff[0];

		norma2 = buff[1];
		norma = sqrt( r[0] * r[0] + norma2 );

		r[0] = ( r[0] > 0 ) ? ( r[0] + norma ) : ( r[0] - norma ) ;
		T[0] = (2.0e+00) / ( (1.0e+00) + norma2 / ( r[0] * r[0] ) ) ;

		if( my_rank == 0 ) cblas_dscal( mloc-1, (1.0e+00)/r[0], A+1, 1);
		else               cblas_dscal( mloc, (1.0e+00)/r[0], A, 1); 
		
		r[0] = ( r[0] > 0 ) ? ( - norma ) : ( + norma ) ;

		if( my_rank == 0 ) for( i = 1 ; i < mloc ; i++ ){ q[i] = - ( A[i] ) * ( T[0] ); }
		else               for( i = 0 ; i < mloc ; i++ ){ q[i] = - ( A[i] ) * ( T[0] ); } 

    		if( my_rank == 0 ) q[0] = (+1.0e+00) - T[0];


	} else {

		if( my_rank == 0 ){
			for( i = 0 ; i < j ; i++ ){ r[i] = A[i+j*lda]; };
			cblas_dtrmv( CblasColMajor, CblasLower, CblasTrans, CblasUnit, j, A, lda, r, 1 );
			cblas_dgemv( CblasColMajor, CblasTrans, mloc-j, j, (+1.0e00), A+j, lda, A+j+j*lda, 1, (+1.0e00), r, 1);
		} else {
			cblas_dgemv( CblasColMajor, CblasTrans, mloc, j, (+1.0e00), A, lda, A+j*lda, 1, (+0.0e00), r, 1);
		}
		MPI_Allreduce( MPI_IN_PLACE, r, j, MPI_DOUBLE, MPI_SUM, mpi_comm);

		cblas_dtrmv( CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit, j, T, ldt, r, 1 );

		if( my_rank == 0 ){
			cblas_dgemv( CblasColMajor, CblasNoTrans, mloc-j, j, (-1.0e00), A+j, lda, r, 1, (+1.0e00), A+j+j*lda, 1);
			cblas_dtrmv( CblasColMajor, CblasLower, CblasNoTrans, CblasUnit, j, A, lda, r, 1 );
			for( i = 0 ; i < j ; i++ ){ r[i] = A[i+j*lda] - r[i]; };
		} else {
			cblas_dgemv( CblasColMajor, CblasNoTrans, mloc, j, (-1.0e00), A, lda, r, 1, (+1.0e00), A+j*lda, 1);
		}

		if( my_rank == 0 ) buff[0] = A[j+j*lda];
		else               buff[0] = 0.0e+00;

		if( my_rank == 0 ) norma2 = cblas_ddot( mloc-j-1, A+j+1+j*lda, 1, A+j+1+j*lda, 1 );
		else               norma2 = cblas_ddot( mloc, A+j*lda, 1, A+j*lda, 1 ); 

		buff[1] = norma2;

		MPI_Allreduce( MPI_IN_PLACE, buff, 2, MPI_DOUBLE, MPI_SUM, mpi_comm);

		if( my_rank == 0 ) r[j] = A[j+j*lda];
		else               r[j] = buff[0];

		norma2 = buff[1];

		norma = sqrt( r[j] * r[j] + norma2 );

		r[j] = ( r[j] > 0 ) ? ( r[j] + norma ) : ( r[j] - norma ) ;

		T[j+j*ldt] = (2.0e+00) / ( (1.0e+00) + norma2 / ( r[j] * r[j] ) ) ;

		if( my_rank == 0 ) cblas_dscal( mloc-j-1, (1.0e+00)/r[j], A+j+1+j*lda, 1);
		else               cblas_dscal( mloc, (1.0e+00)/r[j], A+j*lda, 1);

		if( my_rank == 0 ) r[j] = ( r[j] > 0 ) ? ( - norma ) : ( + norma ) ;

		if( my_rank == 0 ) for( i = j ; i < mloc ; i++ ){ q[i] = - ( A[i+j*lda] ) * ( T[j+j*ldt] ); } 
		else               for( i = 0 ; i < mloc ; i++ ){ q[i] = - ( A[i+j*lda] ) * ( T[j+j*ldt] ); }

    		if( my_rank == 0 ) q[j] = (+1.0e+00) - (T[j+j*ldt]);

		if( my_rank == 0 ) cblas_dgemv( CblasColMajor, CblasTrans, mloc-j, j, (+1.0e00), A+j, lda, q+j, 1, (+0.0e00), T+j*ldt, 1);
		else               cblas_dgemv( CblasColMajor, CblasTrans, mloc, j, (+1.0e00), A, lda, q, 1, (+0.0e00), T+j*ldt, 1);
		MPI_Allreduce( MPI_IN_PLACE, T+j*ldt, j, MPI_DOUBLE, MPI_SUM, mpi_comm);

		cblas_dtrmv( CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, j, T, ldt, T+j*ldt, 1 );

		if( my_rank == 0 ){
			cblas_dgemv( CblasColMajor, CblasNoTrans, mloc-j, j, (-1.0e00), A+j, lda, T+j*ldt, 1, (+1.0e00), q+j, 1);
			cblas_dtrmv( CblasColMajor, CblasLower, CblasNoTrans, CblasUnit, j, A, lda, T+j*ldt, 1 );
			for( i = 0 ; i < j ; i++ ){ q[i] = - T[i+j*ldt]; };
		} else {
			cblas_dgemv( CblasColMajor, CblasNoTrans, mloc, j, (-1.0e00), A, lda, T+j*ldt, 1, (+1.0e00), q, 1);
		}

		if( my_rank == 0 ) { }
		else               for( i = 0 ; i < j+1 ; i++ ){ r[i] = 0e+00; }

		MPI_Allreduce( MPI_IN_PLACE, r, j+1, MPI_DOUBLE, MPI_SUM, mpi_comm);

		if( my_rank == 0 ) {
			cblas_dcopy( j, A+j, lda, T+j*ldt, 1);
			cblas_dgemv( CblasColMajor, CblasTrans, mloc-j-1, j, (+1.0e00), A+j+1, lda, A+j+1+j*lda, 1, (+1.0e00), T+j*ldt, 1);
		} else {
			cblas_dgemv( CblasColMajor, CblasTrans, mloc, j, (+1.0e00), A, lda, A+j*lda, 1, (+0.0e00), T+j*ldt, 1);
		}
		MPI_Allreduce( MPI_IN_PLACE, T+j*ldt, j, MPI_DOUBLE, MPI_SUM, mpi_comm);

		cblas_dscal( j, -T[j+j*ldt], T+j*ldt, 1);

		cblas_dtrmv( CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, j, T, ldt, T+j*ldt, 1 );

	}

	free( buff );

	return 0;

}
