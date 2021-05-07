#include "../c-mpi.h"

int mpi_matvec_manteuffel_M( MPI_Comm mpi_comm, long int m, double *x, double *y ){

	double  tmp=0.0e00, tmp1=0.0e00, tmp2=0.0e00, tmp3=0.0e00;
	int i, mloc, n, iglobal, my_rank, pool_size;

	double alpha = ((double) 4 );

	n = sqrt( m );

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &pool_size);

	mloc = m / pool_size + (( my_rank < m - pool_size*( m / pool_size ) ) ? 1 : 0);
	iglobal = ( m / pool_size ) * my_rank + (( my_rank < m - pool_size*( m / pool_size ) ) ? my_rank : ( m - pool_size * ( m / pool_size ) ));

//	EVEN CASE
	if( my_rank % 2 == 0 ){

		if( my_rank != pool_size-1 ) tmp  = x[mloc-1];
		if( my_rank != 0 )           tmp1 = x[0];

		if( my_rank != pool_size-1 ) MPI_Send(  &tmp, 1, MPI_DOUBLE, my_rank+1, 0, mpi_comm);
		if( my_rank != 0 )           MPI_Send( &tmp1, 1, MPI_DOUBLE, my_rank-1, 0, mpi_comm);

	} else {

		                             MPI_Recv(  &tmp, 1, MPI_DOUBLE, my_rank-1, 0, mpi_comm, MPI_STATUS_IGNORE);
		if( my_rank != pool_size-1 ) MPI_Recv( &tmp1, 1, MPI_DOUBLE, my_rank+1, 0, mpi_comm, MPI_STATUS_IGNORE);


	}

//	ODD CASE
	if( my_rank % 2 == 1 ){

		if( my_rank != pool_size-1 ) tmp2  = x[mloc-1];
		                             tmp3 = x[0];

		if( my_rank != pool_size-1 ) MPI_Send( &tmp2, 1, MPI_DOUBLE, my_rank+1, 0, mpi_comm);
		                             MPI_Send( &tmp3, 1, MPI_DOUBLE, my_rank-1, 0, mpi_comm);

	} else {

		if( my_rank != 0 )           MPI_Recv( &tmp2, 1, MPI_DOUBLE, my_rank-1, 0, mpi_comm, MPI_STATUS_IGNORE);
		if( my_rank != pool_size-1 ) MPI_Recv( &tmp3, 1, MPI_DOUBLE, my_rank+1, 0, mpi_comm, MPI_STATUS_IGNORE);

	}


	if( my_rank == 0 ) {

		y[0] = ( alpha * x[0] ) + ( -1.0e00 * x[1] );

		for(i = 1; i < mloc-1; i++){
 
			if( ( (iglobal+i+1) % (n) ) == 0 ){

				y[i] = ( -1.0e00 * x[i-1] ) + ( alpha * x[i] ) + ( +0.0e00 * x[i+1] ); 					

			} else if( ( ( (iglobal+i+1) % (n) ) == 1 ) && ( i != 1 ) ) {

				y[i] = ( +0.0e00 * x[i-1] ) + ( alpha * x[i] ) + ( -1.0e00 * x[i+1] ); 					

			} else {

				y[i] = ( -1.0e00 * x[i-1] ) + ( alpha * x[i] ) + ( -1.0e00 * x[i+1] ); 

			} 

		}

		if( ( (iglobal+mloc) % (n) ) == 0 ){

			y[mloc-1] = ( -1.0e00 * x[mloc-2] ) + ( alpha * x[mloc-1] ) + ( +0.0e00 * tmp3 );

		} else if( ( (iglobal+mloc) % (n) ) == 1 ){

			y[mloc-1] = ( +0.0e00 * x[mloc-2] ) + ( alpha * x[mloc-1] ) + ( -1.0e00 * tmp3 );

		} else {

			y[mloc-1] = ( -1.0e00 * x[mloc-2] ) + ( alpha * x[mloc-1] ) + ( -1.0e00 * tmp3 );

		}

	}

	if ( ( my_rank != 0 ) && ( my_rank != pool_size-1 ) ) {

		if( ( (iglobal+1) % (n) ) == 0 ){

			y[0] = ( -1.0e00 * tmp2 ) + ( alpha * x[0] ) + ( +0.0e00 * x[1] );

		} else if( ( (iglobal+1) % (n) ) == 1 ){

			y[0] = ( +0.0e00 * tmp2 ) + ( alpha * x[0] ) + ( -1.0e00 * x[1] );

		} else {

			y[0] = ( -1.0e00 * tmp2 ) + ( alpha * x[0] ) + ( -1.0e00 * x[1] );

		}

		for(i = 1; i < mloc-1; i++){
 
			if( ( (iglobal+i+1) % (n) ) == 0 ){

				y[i] = ( -1.0e00 * x[i-1] ) + ( alpha * x[i] ) + ( +0.0e00 * x[i+1] ); 					

			} else if( ( (iglobal+i+1) % (n) ) == 1 ) {

				y[i] = ( +0.0e00 * x[i-1] ) + ( alpha * x[i] ) + ( -1.0e00 * x[i+1] ); 					

			} else {

				y[i] = ( -1.0e00 * x[i-1] ) + ( alpha * x[i] ) + ( -1.0e00 * x[i+1] ); 

			} 

		}

		if( ( (iglobal+mloc) % (n) ) == 0 ){

			y[mloc-1] = ( -1.0e00 * x[mloc-2] ) + ( alpha * x[mloc-1] ) + ( +0.0e00 * tmp3 );

		} else if( ( (iglobal+mloc) % (n) ) == 1 ){

			y[mloc-1] = ( +0.0e00 * x[mloc-2] ) + ( alpha * x[mloc-1] ) + ( -1.0e00 * tmp3 );

		} else {

			y[mloc-1] = ( -1.0e00 * x[mloc-2] ) + ( alpha * x[mloc-1] ) + ( -1.0e00 * tmp3 );

		}

	}

	if ( my_rank == pool_size-1 ) {

		if( ((iglobal+1)%(n)) == 0 ){			

			y[0] = ( -1.0e00 * tmp ) + ( alpha * x[0] ) + ( +0.0e00 * x[1] );

		} else if( ((iglobal+1)%(n)) == 1 ){

			y[0] = ( +0.0e00 * tmp ) + ( alpha * x[0] ) + ( -1.0e00 * x[1] );

		} else {

			y[0] = ( -1.0e00 * tmp ) + ( alpha * x[0] ) + ( -1.0e00 * x[1] );

		}

		for(i = 1; i < mloc-1; i++){
 
			if( ((iglobal+i+1)%(n)) == 0 ){

				y[i] = ( -1.0e00 * x[i-1] ) + ( alpha * x[i] ) + ( +0.0e00 * x[i+1] ); 					

			} else if( ((iglobal+i+1)%(n)) == 1 ) {

				y[i] = ( +0.0e00 * x[i-1] ) + ( alpha * x[i] ) + ( -1.0e00 * x[i+1] ); 					

			} else {

				y[i] = ( -1.0e00 * x[i-1] ) + ( alpha * x[i] ) + ( -1.0e00 * x[i+1] ); 

			} 

		}

		y[mloc-1] = ( -1.0e00 * x[mloc-2] ) + ( alpha * x[mloc-1] );

	}

	return 0;

}
