#include "../c-mpi.h"

int mpi_orth_mgs_lvl1_manysynch_append1(int mloc, int j, double *A, int lda, double *r, MPI_Comm mpi_comm){

//	There are ( j + 1 ) synchronizations per step.

	int i;

	for ( i = 0; i < j; i++) {
		r[i] = cblas_ddot(mloc,&(A[i*mloc]),1,&(A[j*mloc]),1) ;
		MPI_Allreduce( MPI_IN_PLACE, &(r[i]), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		cblas_daxpy(mloc,-r[i],&(A[i*mloc]),1,&(A[j*mloc]),1);
	}
	r[j] = cblas_ddot(mloc,&(A[j*mloc]),1,&(A[j*mloc]),1);
	MPI_Allreduce( MPI_IN_PLACE, &(r[j]), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	r[j] = sqrt(r[j]);
	cblas_dscal(mloc,1/r[j],&(A[j*mloc]),1);

	return 0;

}
