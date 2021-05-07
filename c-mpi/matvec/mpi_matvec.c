#include "../c-mpi.h"

int mpi_matvec( MPI_Comm mpi_comm, long int m, double *x, double *y ){

	double  tmp=0.0e00, tmp1=0.0e00, tmp2=0.0e00, tmp3=0.0e00;
	int i, mloc, iglobal0, my_rank, pool_size;

//	double alpha= ((double) m )*1000000000000e+00;
//	double alpha= ((double) m )*1000000000e+00;
//	double alpha= ((double) m )*1000e+00;
//	double alpha= ((double) m );
//	double alpha= ((double) 1 );
	double alpha= ((double) 4 );
//	double alpha= ((double) 1e-16 );
//	double alpha= ((double) 1e-30 );
//	double alpha= ((double) 0e00 );

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &pool_size);

	mloc     =   m / pool_size             + (( my_rank < m - pool_size*( m / pool_size ) ) ? 1 : 0);
	iglobal0 = ( m / pool_size ) * my_rank + (( my_rank < m - pool_size*( m / pool_size ) ) ? my_rank : ( m - pool_size * ( m / pool_size ) ));

//	EVEN CASE
	if( my_rank % 2 == 0 ){

		if( my_rank != pool_size-1 ) tmp  = x[mloc-1];
		if( my_rank != 0 )           tmp1 = x[0];

		if( my_rank != pool_size-1 ) MPI_Send(  &tmp, 1, MPI_DOUBLE, my_rank+1, 0, mpi_comm);
		if( my_rank != 0 )           MPI_Send( &tmp1, 1, MPI_DOUBLE, my_rank-1, 0, mpi_comm);

	} else {

		                             MPI_Recv(  &tmp, 1, MPI_DOUBLE, my_rank-1, 0, mpi_comm, MPI_STATUS_IGNORE);
		if( my_rank != pool_size-1 ) MPI_Recv( &tmp1, 1, MPI_DOUBLE, my_rank+1, 0, mpi_comm, MPI_STATUS_IGNORE);

		if( my_rank != pool_size-1 ) {

			y[0] = ( iglobal0 * tmp ) + ( ( alpha + iglobal0 ) * x[0] ) + ( ( iglobal0 + 1 ) * x[1] );

			for(i = 1; i < mloc-1; i++) y[i] = ( ( m - ( iglobal0 + i ) ) * x[i-1] ) + ( ( alpha + iglobal0 + i ) * x[i] ) + ( ( iglobal0 + i + 1 ) * x[i+1] );

			y[mloc-1] = ( ( m - ( iglobal0 + mloc - 1 ) ) * x[mloc-2] ) + ( ( alpha + iglobal0 + mloc - 1 ) * x[mloc-1] ) + ( ( iglobal0 + mloc ) * tmp1 );

		}

		if ( my_rank == pool_size-1 ) {

			y[0] = ( iglobal0 * tmp ) + ( ( alpha + iglobal0 ) * x[0] ) + ( ( iglobal0 + 1 ) * x[1] );

			for(i = 1; i < mloc-1; i++) y[i] = ( ( m - ( iglobal0 + i ) )* x[i-1] ) + ( ( alpha + iglobal0 + i ) * x[i] ) + ( ( iglobal0 + i + 1 ) * x[i+1] );

			y[mloc-1] = ( ( m - ( iglobal0 + mloc - 1 ) ) * x[mloc-2] ) + ( ( alpha + iglobal0 + mloc - 1 ) * x[mloc-1] );

		}
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

		if( my_rank == 0 ) {

			y[0] = ( alpha + iglobal0 ) * x[0] + x[1];

			for(i = 1; i < mloc-1; i++) y[i] = ( ( m - ( iglobal0 + i ) ) * x[i-1] ) + ( ( alpha + iglobal0 + i ) * x[i] ) + ( ( iglobal0 + i + 1 ) * x[i+1] );

			y[mloc-1] = ( ( iglobal0 + mloc - 1 ) * x[mloc-2] ) + ( ( alpha + iglobal0 + mloc - 1 ) * x[mloc-1] ) + ( ( iglobal0 + mloc ) * tmp3 );

		}

		if ( ( my_rank != 0 ) && ( my_rank != pool_size-1 ) ) {

			y[0] = ( ( m - iglobal0 ) * tmp2 ) + ( ( alpha + iglobal0 ) * x[0] ) + ( ( iglobal0 + 1 ) * x[1] );

			for(i = 1; i < mloc-1; i++) y[i] = ( ( m - ( iglobal0 + i ) ) * x[i-1] ) + ( ( alpha + iglobal0 + i ) * x[i] ) + ( ( iglobal0 + i + 1 ) * x[i+1] );

			y[mloc-1] = ( ( m - ( iglobal0 + mloc - 1 ) ) * x[mloc-2] ) + ( ( alpha + iglobal0 + mloc - 1 ) * x[mloc-1] ) + ( ( iglobal0 + mloc ) * tmp3 );

		}

		if ( my_rank == pool_size-1 ) {

			y[0] = ( ( m - iglobal0 ) * tmp2 ) + ( ( alpha + iglobal0 ) * x[0] ) + ( ( iglobal0 + 1 ) * x[1] );

			for(i = 1; i < mloc-1; i++) y[i] = ( ( m - ( iglobal0 + i ) ) * x[i-1] ) + ( ( alpha + iglobal0 + i ) * x[i] ) + ( ( iglobal0 + i + 1 ) * x[i+1] );

			y[mloc-1] = ( ( m - ( iglobal0 + mloc - 1 ) )* x[mloc-2] ) + ( ( alpha + iglobal0 + mloc - 1 ) * x[mloc-1] );

		}

	} 

	return 0;

}
