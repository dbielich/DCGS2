#include "../c-mpi.h"

int mpi_orth_dcgs2_onesynch_cleanup_append1_qr( int mloc, int j, double *Q, int ldq, double *R, int ldr, double *work, MPI_Comm mpi_comm){

	int i;
	double tmp;

	cblas_dgemv(CblasColMajor, CblasTrans, mloc, j-1, 1.0e0, Q, ldq, &(Q[(j-1)*ldq]), 1, 0.0e0, &(work[0]), 1);
	work[j] = cblas_ddot(ldq,Q+(j-1)*ldq,1,Q+(j-1)*ldq,1);

	MPI_Allreduce( MPI_IN_PLACE, &(work[0]), j+1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	tmp = 0.0e+00; for(i=0;i<j-1;i++) tmp = tmp + ( work[i] * work[i] );
        R[(j-1)+(j-1)*ldr] = sqrt( work[j] - tmp );
	cblas_dgemv(CblasColMajor, CblasNoTrans, mloc, 1, -1.0e0,  Q, ldq, &(work[0]), 1, 1.0e0, &(Q[(j-1)*ldq]), 1);
        for(i=0;i<j-1;i++) R[i+j*ldr] = R[i+(j-1)*ldr] + work[i];
	cblas_dscal(ldq,1/R[j-1+(j-1)*ldr],Q+(j-1)*ldq,1);

	return 0;

}
