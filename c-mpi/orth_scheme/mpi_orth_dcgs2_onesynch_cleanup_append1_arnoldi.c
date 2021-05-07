#include "../c-mpi.h"

int mpi_orth_dcgs2_onesynch_cleanup_append1_arnoldi( int mloc, int j, double *Q, int ldq, double *H, int ldh, double *work, MPI_Comm mpi_comm){

	int i;
	double BETA, ALPHA;
	double *work1, *work2;

	work1 = &(work[0]);
	work2 = &(work[j]);

	cblas_dgemv(CblasColMajor, CblasTrans, mloc, j+1, 1.0e0,  Q, mloc, &(Q[j*mloc]), 1, 0.0e0, &(work1[0]), 1);
	MPI_Allreduce( MPI_IN_PLACE, &(work1[0]), j+1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	BETA  = work2[0];
	ALPHA = 0.0e00; for(i=0;i<j;i++) ALPHA = ALPHA + work1[i] * work1[i];

	for(i=0;i<j;i++) H[i+(j-1)*ldh] = H[i+(j-1)*ldh] + work1[i];
	H[j+(j-1)*ldh] = sqrt(BETA - ALPHA);

	for(i=0;i<j;i++) work1[i] = work1[i] / H[j+(j-1)*ldh];
	cblas_dgemv(CblasColMajor, CblasNoTrans, mloc, j, (-1.0e0), Q, mloc, &(work1[0]), 1, 1/H[j+(j-1)*ldh], &(Q[j*mloc]), 1);


	return 0;

}
