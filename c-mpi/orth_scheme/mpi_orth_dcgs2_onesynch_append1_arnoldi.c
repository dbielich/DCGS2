#include "../c-mpi.h"

int mpi_orth_dcgs2_onesynch_append1( int mloc, int j, double *Q, int ldq, double *H, int ldh, double *K, int ldk, double *work, MPI_Comm mpi_comm){

	double *work1, *work2;
	double tmp, ALPHA, BETA;
	int i;

	if( j == 0 ){

		H[0] = cblas_ddot(ldq,Q,1,Q,1);
		MPI_Allreduce( MPI_IN_PLACE, &(H[0]), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		H[0] = sqrt(H[0]);
		cblas_dscal(ldq,1/H[0],Q,1);

	}

	if ( j == 1 ){

		work[0] = cblas_ddot(ldq,Q,1,Q+ldq,1);
		work[1] = cblas_ddot(ldq,Q,1,Q,1);

		MPI_Allreduce( MPI_IN_PLACE, &(work[0]), 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		H[1] = sqrt( work[1] );
		K[0] = work[0] / ( H[1] * H[1] );
		for(i=0;i<ldq;i++) Q[i+ldq] = ( Q[i+ldq] / H[1] ) - Q[i] * K[0];			

	}

	if ( j > 1 ){

		work1 = &(work[0]);
		work2 = &(work[j]);

		cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, j, 2, mloc, 1.0e0, Q, ldq, &(Q[(j-1)*ldq]), ldq, 0.0e0, &(work[0]), j);
		MPI_Allreduce( MPI_IN_PLACE, &(work[0]), 2*j, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		BETA  = work1[j-1];
		ALPHA = 0.0e00;	for(i=0;i<j-1;i++) ALPHA = ALPHA + work1[i] * work1[i];

		for(i=0;i<j-1;i++) H[i+(j-2)*ldh] = K[i+(j-2)*ldk] + work1[i];
		H[(j-1)+(j-2)*ldh] = sqrt( BETA - ALPHA );
	
		tmp = 0.0e00; for(i=0;i<j-1;i++) tmp = tmp + work1[i] * work2[i];
		tmp = ( work2[j-1] - tmp ) / ( H[(j-1)+(j-2)*ldh] * H[(j-1)+(j-2)*ldh] );

		for(i=0;i<j-1;i++) K[i+(j-1)*ldh] = work2[i] / H[(j-1)+(j-2)*ldh];
		for(i=0;i<j-1;i++) work2[i]       = work2[i] / H[(j-1)+(j-2)*ldh];
		for(i=0;i<j-1;i++) work1[i]       = work1[i] / H[(j-1)+(j-2)*ldh];

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ldq, 2, j-1, -1.0e00, Q, ldq, work, j, 1/H[(j-1)+(j-2)*ldh], &(Q[(j-1)*ldq]), ldq);
		cblas_daxpy( ldq, -tmp, &(Q[(j-1)*ldq]), 1, &(Q[j*ldq]), 1);

		for(i=0;i<j-1;i++) work1[i] = work1[i] * H[(j-1)+(j-2)*ldh];

		cblas_dgemv(CblasColMajor, CblasNoTrans, j, j-1, 1.0e0,  H, ldh, &(work1[0]), 1, 0.0e0, &(work2[0]), 1);

//		work2[0] = 0.0e+00; for(i=1;i<j-1;i++) work2[i] = work1[i];
//		for(i=0;i<j-1;i++) work2[0] += H[i+ldh] * work1[i];
//		cblas_dtrmv( CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, j-1, H+1, ldh, &(work2[1]), 1 );

		for(i=0;i<j-1;i++) K[i+(j-1)*ldk] = K[i+(j-1)*ldk] - ( work2[i] / H[(j-1)+(j-2)*ldh] );
		K[(j-1)+(j-1)*ldk] = tmp - ( work2[(j-1)] / H[(j-1)+(j-2)*ldh] );

	}

	return 0;

}
