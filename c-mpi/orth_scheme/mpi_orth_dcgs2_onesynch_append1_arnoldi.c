#include "../c-mpi.h"

int mpi_orth_dcgs2_onesynch_append1_arnoldi( int mloc, int j, double *Q, int ldq, double *H, int ldh, double *work, MPI_Comm mpi_comm){

	double *work1, *work2;
	double tmp, ALPHA, BETA;
	int i;

	if ( j == 1 ){


		work[0] = cblas_ddot(mloc,Q,1,Q+mloc,1);
		work[1] = cblas_ddot(mloc,Q,1,Q,1);

		MPI_Allreduce( MPI_IN_PLACE, &(work[0]), 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		H[1] = sqrt( work[1] );
		H[0] = work[0] / ( H[1] * H[1] );
		for(i=0;i<mloc;i++) Q[i+mloc] = ( Q[i+mloc] / H[1] ) - Q[i] * H[0];			

	}

	if ( j > 1 ){

		work1 = &(work[0]);
		work2 = &(work[j]);

		cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, j, 2, mloc, 1.0e0, Q, mloc, &(Q[(j-1)*mloc]), mloc, 0.0e0, &(work[0]), j);
		MPI_Allreduce( MPI_IN_PLACE, &(work[0]), 2*j, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		BETA  = work1[j-1];
		ALPHA = 0.0e00;	for(i=0;i<j-1;i++) ALPHA = ALPHA + work1[i] * work1[i];

		for(i=0;i<j-1;i++) H[i+(j-2)*ldh] = H[i+(j-2)*ldh] + work1[i];
		H[(j-1)+(j-2)*ldh] = sqrt( BETA - ALPHA );

		tmp = 0.0e00; for(i=0;i<j-1;i++) tmp = tmp + work1[i] * work2[i];
		tmp = ( work2[j-1] - tmp ) / ( H[(j-1)+(j-2)*ldh] * H[(j-1)+(j-2)*ldh] );

		for(i=0;i<j-1;i++) H[i+(j-1)*ldh] = work2[i] / H[(j-1)+(j-2)*ldh];
		for(i=0;i<j-1;i++) work2[i]       = work2[i] / H[(j-1)+(j-2)*ldh];
		for(i=0;i<j-1;i++) work1[i]       = work1[i] / H[(j-1)+(j-2)*ldh];

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, mloc, 2, j-1, -1.0e00, Q, mloc, work, j, 1/H[(j-1)+(j-2)*ldh], &(Q[(j-1)*mloc]), mloc);
		cblas_daxpy( mloc, -tmp, &(Q[(j-1)*mloc]), 1, &(Q[j*mloc]), 1);

		for(i=0;i<j-1;i++) work1[i] = work1[i] * H[(j-1)+(j-2)*ldh];

//		cblas_dgemv(CblasColMajor, CblasNoTrans, j, j-1, 1.0e0,  H, ldh, &(work1[0]), 1, 0.0e0, &(work2[0]), 1);
		for(i=0;i<j-1;i++) work2[i+1] = work1[i];
		cblas_dtrmv( CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, j-1, H+1, ldh, &(work2[1]), 1 );
		work2[0] = 0.0e+00; for(i=0;i<j-1;i++) work2[0] += H[i*ldh] * work1[i];

		for(i=0;i<j-1;i++) H[i+(j-1)*ldh] = H[i+(j-1)*ldh] - ( work2[i] / H[(j-1)+(j-2)*ldh] );
		H[(j-1)+(j-1)*ldh] = tmp - ( work2[(j-1)] / H[(j-1)+(j-2)*ldh] );

	}

	return 0;

}
