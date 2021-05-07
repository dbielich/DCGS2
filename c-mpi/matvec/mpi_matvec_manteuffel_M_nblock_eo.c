#include "../c-mpi.h"

int mpi_matvec_manteuffel_M_nblock_eo( MPI_Comm mpi_comm, long int m, double *x, double *y, double *x_notlocal ){

	int n, i, j, k, mloc, iglobal, my_rank, pool_size; 

	n = sqrt( m );

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &pool_size);

	mloc = m / pool_size + (( my_rank < m - pool_size*( m / pool_size ) ) ? 1 : 0);
	iglobal = ( m / pool_size ) * my_rank + (( my_rank < m - pool_size*( m / pool_size ) ) ? my_rank : ( m - pool_size * ( m / pool_size ) ));

	if( my_rank % 2 == 0 ){

		if( pool_size != 1 ){
			if( my_rank != pool_size-1 ) MPI_Send( &(x[mloc-n]), n, MPI_DOUBLE, my_rank+1, 0, mpi_comm);
			if( my_rank != 0           ) MPI_Send( &(x[0]), n, MPI_DOUBLE, my_rank-1, 0, mpi_comm);
		}

	} else {

		if( pool_size != 1 ){
			if( my_rank != 0           ) MPI_Recv( &(x_notlocal[0]), n, MPI_DOUBLE, my_rank-1, 0, mpi_comm, MPI_STATUS_IGNORE); 
			if( my_rank != pool_size-1 ) MPI_Recv( &(x_notlocal[n]), n, MPI_DOUBLE, my_rank+1, 0, mpi_comm, MPI_STATUS_IGNORE); 
		}

	}


	if( my_rank % 2 == 1 ){

		if( pool_size != 1 ){
			if( my_rank != pool_size-1 ) MPI_Send( &(x[mloc-n]), n, MPI_DOUBLE, my_rank+1, 0, mpi_comm);
			if( my_rank != 0           ) MPI_Send( &(x[0]), n, MPI_DOUBLE, my_rank-1, 0, mpi_comm);
		}

	} else {

		if( pool_size != 1 ){
			if( my_rank != 0           ) MPI_Recv( &(x_notlocal[0]), n, MPI_DOUBLE, my_rank-1, 0, mpi_comm, MPI_STATUS_IGNORE); 
			if( my_rank != pool_size-1 ) MPI_Recv( &(x_notlocal[n]), n, MPI_DOUBLE, my_rank+1, 0, mpi_comm, MPI_STATUS_IGNORE); 
		}

	}

	if ( my_rank == 0 ){

		if( mloc > 2*n ){
		
			for( i=0;i<n;i++ ) y[i] += ( -1.0e00 * x[i+n] );
			if( pool_size == 1 ){ 
				for( i=n;i<mloc-n;i++ )    y[i] += ( -1.0e00 * x[i-n] ) + ( -1.0e00 * x[i+n] );
				for( i=mloc-n;i<mloc;i++ ) y[i] += ( -1.0e00 * x[i-n] );
			} else { 
				for( i=n;i<mloc-n;i++ ) y[i] += ( -1.0e00 * x[i-n] ) + ( -1.0e00 * x[i+n] );
				j=0; for( i=mloc-n;i<mloc;i++ ){ y[i] += ( -1.0e00 * x[i-n] ) + ( -1.0e00 * x_notlocal[j+n] ); j++; }
			}
		} else {
			for( i=0;i<mloc-n;i++ ) y[i] += ( -1.0e00 * x[i+n] );
			j=0; for( i=mloc-n;i<n;i++ ){ y[i] += ( -1.0e00 * x_notlocal[j+n] ); j++; }

			if( pool_size == 1 ){ 
				for( i=n;i<mloc-n;i++ )    y[i] += ( -1.0e00 * x[i-n] ) + ( -1.0e00 * x[i+n] );
				for( i=mloc-n;i<mloc;i++ ) y[i] += ( -1.0e00 * x[i-n] );
			} else { 
				for( i=n;i<mloc;i++ ){ y[i] += ( -1.0e00 * x[i-n] ) + ( -1.0e00 * x_notlocal[j+n] ); j++; }
			}
		}
		
	} else if ( ( my_rank == pool_size-1 ) && ( my_rank != 0 ) ){ 

		if ( mloc > 2*n ){

			j=0; for( i=0;i<n;i++ ){ y[i] += ( -1.0e00 * x_notlocal[j] ) + ( -1.0e00 * x[i+n] ); j++; }
			for( i=n;i<mloc-n;i++ ) y[i] += ( -1.0e00 * x[i-n] ) + ( -1.0e00 * x[i+n] );
			for( i=mloc-n;i<mloc;i++ ) y[i] += ( -1.0e00 * x[i-n] );

		} else {

			j=0; for( i=0;i<mloc-n;i++ ){ y[i] += ( -1.0e00 * x_notlocal[j] ) + ( -1.0e00 * x[i+n] ); j++; }
			for( i=mloc-n;i<n;i++ ){ y[i] += ( -1.0e00 * x_notlocal[j] ); j++; }
			for( i=n;i<mloc;i++ ) y[i] += ( -1.0e00 * x[i-n] );

		}

	} else {

		if ( mloc > 2*n ){

			j=0; for( i=0;i<n;i++ ){ y[i] += ( -1.0e00 * x_notlocal[j] ) + ( -1.0e00 * x[i+n] ); j++; }
			for( i=n;i<mloc-n;i++ ) y[i] += ( -1.0e00 * x[i-n] ) + ( -1.0e00 * x[i+n] );
			j=0; for( i=mloc-n;i<mloc;i++ ){ y[i] += ( -1.0e00 * x[i-n] ) + ( -1.0e00 * x_notlocal[j+n] ); j++; }

		} else {

			j=0; for( i=0;i<mloc-n;i++ ){ y[i] += ( -1.0e00 * x_notlocal[j] ) + ( -1.0e00 * x[i+n] ); j++; }
			k=0; for( i=mloc-n;i<n;i++ ){ y[i] += ( -1.0e00 * x_notlocal[j] ) + ( -1.0e00 * x_notlocal[k+n] ); j++; k++; }
			for( i=n;i<mloc;i++ ){ y[i] += ( -1.0e00 * x[i-n] ) + ( -1.0e00 * x_notlocal[k+n] ); k++; }

		}

	}

	return 0;

}
