#include "../c-mpi.h"

int mpi_check_qq_orth( MPI_Comm mpi_comm, double *orthlevel, int mloc, int n, double *Q, int ldq  ){

	/* Be careful after this routine orthlevel is defined only on one process: process 0! */
	/* Data distribution of the matrix Q is by row */

	int j, my_rank;
	double *temp_buf = NULL;

	(*orthlevel)=0xDEADBEEF;

	MPI_Comm_rank( mpi_comm, &my_rank);

	temp_buf = (double *)malloc(n*n*sizeof(double)) ;
	for (j = 0; j < n*n; j++) temp_buf[j]=0xDEADBEEF;

	cblas_dsyrk( CblasColMajor, CblasUpper, CblasTrans, n, mloc, 1.0e+00, Q, ldq, 0e+00, temp_buf, n); 
	if (my_rank==0) MPI_Reduce( MPI_IN_PLACE, temp_buf, n*n, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);
	else MPI_Reduce( temp_buf, NULL, n*n, MPI_DOUBLE, MPI_SUM, 0, mpi_comm);
	if (my_rank==0) for ( j = 0; j < n; j++) temp_buf[j*(n+1)] -= 1.0e+00;
	if (my_rank==0) (*orthlevel) = LAPACKE_dlansy_work( LAPACK_COL_MAJOR, 'F', 'U', n, temp_buf, n, NULL );

	free(temp_buf);

	return 0;
}
