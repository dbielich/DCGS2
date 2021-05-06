#include "../c-mpi.h"

int mpi_orth_hh_lvl1_manysynch_append1( MPI_Comm mpi_comm, int mloc, int j, double *A, int lda, double *tau, double *q, double *r ){

//	There are 2 * ( j + 1 ) synchronizations per step (at step zero there is only one needed)
//
//	j is the index of the column you want to add in C, so 
//		if j=0, you have n=1 column
//		if j=1, you have n=2 columns
//	j is also the number of columns before the current column
//
//
//	# flops = ( function of m and j ) ( on proc NOT 0, since they are doing the most work )
//
//	# comm = ( j ) * MPI_Allreduce( 1 ) + ( 1 ) * MPI_Allreduce( 2 ) + ( j ) * MPI_Allreduce( 1 ) + ( 1 ) * MPI_Allreduce( j+1 ) 
//
//	# comm = ( 2 * j + 2 ) * ( latency ) + ( 3 * j + 3 ) * ( inverse of bandwidth )

	double norma, norma2;
	double dtmp;
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
		tau[0] = (2.0e+00) / ( (1.0e+00) + norma2 / ( r[0] * r[0] ) ) ;

		if( my_rank == 0 ) cblas_dscal( mloc-1, (1.0e+00)/r[0], A+1, 1);
		else               cblas_dscal( mloc, (1.0e+00)/r[0], A, 1); 
		
		r[0] = ( r[0] > 0 ) ? ( - norma ) : ( + norma ) ;

		if( my_rank == 0 ) for( i = 1 ; i < mloc ; i++ ){ q[i] = - ( A[i] ) * ( tau[0] ); }
		else               for( i = 0 ; i < mloc ; i++ ){ q[i] = - ( A[i] ) * ( tau[0] ); } 

    		if( my_rank == 0 ) q[0] = (+1.0e+00) - tau[0];

	} else {

		for( i = 0 ; i < j ; i++ ){

			if( my_rank == 0 ) dtmp = A[i+j*lda] + cblas_ddot( mloc-i-1, A+i+1+i*lda, 1, A+i+1+j*lda, 1 );
			else               dtmp = cblas_ddot( mloc, A+i*lda, 1, A+j*lda, 1 );

			MPI_Allreduce( MPI_IN_PLACE, &dtmp, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);

			dtmp *= -(tau[i]) ;

			if( my_rank == 0 ) cblas_daxpy( mloc-i-1, dtmp, A+i+1+i*lda, 1, A+i+1+j*lda, 1 );
			else               cblas_daxpy( mloc, dtmp, A+i*lda, 1, A+j*lda, 1 );

			if( my_rank == 0 ) A[i+j*lda] = A[i+j*lda] + dtmp;

		}

		if( my_rank == 0 ) buff[0] = A[j+j*lda];
		else               buff[0] = 0.0e+00;

		if( my_rank == 0 ) norma2 = cblas_ddot( mloc-j-1, A+j+1+j*lda, 1, A+j+1+j*lda, 1 );
		else               norma2 = cblas_ddot( mloc, A+j*lda, 1, A+j*lda, 1 ); 

		buff[1] = norma2;

		MPI_Allreduce( MPI_IN_PLACE, buff, 2, MPI_DOUBLE, MPI_SUM, mpi_comm);

		if( my_rank == 0 ) dtmp = A[j+j*lda];
		else               dtmp = buff[0];

		norma2 = buff[1];

		norma = sqrt( dtmp * dtmp + norma2 );

		dtmp = ( dtmp > 0 ) ? ( dtmp + norma ) : ( dtmp - norma ) ;

		tau[j] = (2.0e+00) / ( (1.0e+00) + norma2 / ( dtmp * dtmp ) ) ;

		if( my_rank == 0 ) cblas_dscal( mloc-j-1, (1.0e+00)/dtmp, A+j+1+j*lda, 1);
		else               cblas_dscal( mloc, (1.0e+00)/dtmp, A+j*lda, 1);

		if( my_rank == 0 ) A[j+j*lda] = ( dtmp > 0 ) ? ( - norma ) : ( + norma ) ;

		if( my_rank == 0 ) for( i = j ; i < mloc ; i++ ){ q[i] = - ( A[i+j*lda] ) * ( tau[j] ); } 
		else               for( i = 0 ; i < mloc ; i++ ){ q[i] = - ( A[i+j*lda] ) * ( tau[j] ); }

    		if( my_rank == 0 ) q[j] = (+1.0e+00) - (tau[j]);

		for( i = j-1 ; i >= 0 ; i-- ){

			if( my_rank == 0 ) dtmp = cblas_ddot( mloc-i-1, A+i+1+i*lda, 1, q+i+1, 1 );
			else               dtmp = cblas_ddot( mloc, A+i*lda, 1, q, 1 );

			MPI_Allreduce( MPI_IN_PLACE, &dtmp, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);

			dtmp *= -(tau[i]) ;

			if( my_rank == 0 ) cblas_daxpy( mloc-i-1, dtmp, A+i+1+i*lda, 1, q+i+1, 1 );
			else               cblas_daxpy( mloc, dtmp, A+i*lda, 1, q, 1 );

			if( my_rank == 0 ) q[i] = dtmp;

		}

		if( my_rank == 0 ) cblas_dcopy( j+1, A+j*lda, 1, r, 1 );
		else               for( i = 0 ; i < j+1 ; i++ ){ r[i] = 0e+00; }

		MPI_Allreduce( MPI_IN_PLACE, r, j+1, MPI_DOUBLE, MPI_SUM, mpi_comm);

	}

	free( buff );

	return 0;

}
