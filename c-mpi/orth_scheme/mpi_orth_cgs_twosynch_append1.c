#include "../c-mpi.h"

int mpi_orth_cgs_twosynch_append1(int mloc, int i, double *A, int lda, double *r, MPI_Comm mpi_comm){


	if( i == 0){

		r[0] = cblas_ddot(mloc,A,1,A,1);
		MPI_Allreduce( MPI_IN_PLACE, &(r[0]), 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
		r[0] = sqrt(r[0]);
		cblas_dscal(mloc,1/r[0],A,1);

	} else {

		cblas_dgemv(CblasColMajor, CblasTrans, mloc, i, 1.0e0,  A, mloc, &(A[i*mloc]), 1,
			0.0e0, &(r[0]), 1);
		MPI_Allreduce( MPI_IN_PLACE, &(r[0]), i, MPI_DOUBLE, MPI_SUM, mpi_comm);
		cblas_dgemv(CblasColMajor, CblasNoTrans, mloc, i, (-1.0e0),  A, mloc, &(r[0]), 1,
			1.0e0, &(A[i*mloc]), 1);
	
		r[i] = cblas_ddot(mloc,&(A[i*mloc]),1,&(A[i*mloc]),1);
		MPI_Allreduce( MPI_IN_PLACE, &(r[i]), 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
		r[i] = sqrt(r[i]);

		cblas_dscal(mloc,1/r[i],&(A[i*mloc]),1);

	}

	return 0;

}
