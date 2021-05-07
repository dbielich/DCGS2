#include "../c-mpi.h"

int mpi_orth_dcgs2_hrt_onesynch_cleanup_append1_arnoldi( int mloc, int j, double *W, int ldw, double *U, int ldu, double *Q, int ldq, double *H, int ldh, double *C, int ldc, double *rho, double *work, MPI_Comm mpi_comm){

	int i;

	H[(j-1)+(j-2)*ldh] = cblas_ddot(mloc,&(W[(j-1)*ldw]),1,&(W[(j-1)*ldw]),1);
	cblas_dgemv(CblasColMajor, CblasTrans, mloc, j-1, 1.0e0,  Q, ldq, &(W[j*ldw]), 1, 0.0e0, &(C[(j-1)*ldc]), 1);
	cblas_dgemv(CblasColMajor, CblasNoTrans, mloc, j-1, -1.0e0,  Q, ldq, &(C[(j-1)*ldc]), 1, 1.0e0, &(W[j*ldw]), 1);
	MPI_Allreduce( MPI_IN_PLACE, &(H[(j-1)+(j-2)*ldh]), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	H[(j-1)+(j-2)*ldh] = sqrt( H[(j-1)+(j-2)*ldh] );
	for( i=0;i<mloc;i++) Q[i+(j-1)*ldq] = W[i+(j-1)*ldw] / H[(j-1)+(j-2)*ldh];
	for( i=0;i<j-1;i++) H[i+(j-1)*ldh] += C[i+(j-1)*ldc];
	H[j+(j-1)*ldh] = cblas_ddot(mloc,&(W[i+j*ldw]),1,&(W[i+j*ldw]),1);
	MPI_Allreduce( MPI_IN_PLACE, &(H[j+(j-1)*ldh]), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce( MPI_IN_PLACE, &(C[(j-1)*ldc]), j-1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	H[j+(j-1)*ldh] = sqrt( H[j+(j-1)*ldh] );
	for( i=0;i<mloc;i++) Q[i+j*ldq] = W[i+j*ldw] / H[j+(j-1)*ldh];
			
	return 0;

}
