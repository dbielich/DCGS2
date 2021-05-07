#include "../c-mpi.h"

int mpi_orth_dcgs2_hrt_onesynch_append1_arnoldi( int mloc, int j, double *W, int ldw, double *U, int ldu, double *Q, int ldq, double *H, int ldh, double *C, int ldc, double *rho, double *work, MPI_Comm mpi_comm){

	int i;

	if ( j == 1 ){

		double beta;

		beta = H[0];
		H[0] = cblas_ddot(mloc,&(W[(0)*ldw]),1,&(W[1*ldw]),1);
		MPI_Allreduce( MPI_IN_PLACE, &(H[0]), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		H[0] = H[0] / ( beta * beta );
		for( i=0;i<mloc;i++ ) U[i+(0)*ldu] = W[i+(0)*ldw] / beta;
		cblas_dscal(mloc,1/beta,&(W[1*ldw]),1);
		cblas_daxpy( mloc, -H[0], &(U[0]), 1, &(W[1*ldw]), 1 );

	}

	if ( j == 2 ){

		work[0] = cblas_ddot(mloc,&(W[(1)*ldw]),1,&(W[1*ldw]),1);
		work[1] = cblas_ddot(mloc,&(U[(0)*ldu]),1,&(W[2*ldw]),1);
		work[2] = cblas_ddot(mloc,&(W[1*ldw]),1,&(W[2*ldw]),1);
		work[3] = 0.0e00; for( i=0;i<mloc;i++ ) work[3] += U[i+(0)*ldu] * W[i+1*ldw];
		MPI_Allreduce( MPI_IN_PLACE, work, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		C[0] = work[3];
		rho[0] = sqrt( work[0] );
		H[0+1*ldh] = work[1];
		H[1+1*ldh] = work[2];
		for( i=0;i<mloc;i++ ) U[i+1*ldu] = W[i+1*ldw] / rho[0];
		for( i=0;i<mloc;i++ ) W[i+2*ldw] = W[i+2*ldw] / rho[0];
		for( i=0;i<1;i++ )    H[i+1*ldh] = H[i+1*ldh] / rho[0];
		H[1+1*ldh] = H[1+1*ldh] / ( rho[0] * rho[0] );
		for( i=0;i<mloc;i++ ) W[i+2*ldw] = W[i+2*ldw] - U[i+(0)*ldu]*H[(0)+1*ldh] - U[i+(1)*ldu]*H[(1)+1*ldh];
		for( i=0;i<mloc;i++ ) W[i+1*ldw] = W[i+1*ldw] - U[i+(0)*ldu]*C[0];
		H[0] += C[0];
		for( i=0;i<mloc;i++ ) Q[i+(0)*ldq] = U[i+(0)*ldu];

	}

	if ( j >= 3 ){

		cblas_dgemv(CblasColMajor, CblasTrans, mloc, (j-2), 1.0e0,  Q, ldq, W+(j-1)*ldw, 1, 0.0e0, &(work[0]), 1);
		cblas_dgemv(CblasColMajor, CblasTrans, mloc, (j-2), 1.0e0,  Q, ldq, &(W[j*ldw]), 1, 0.0e0, &(work[j-2]), 1);
		work[2*j-4] = cblas_ddot(mloc,&(U[(j-2)*ldu]),1,&(W[(j-1)*ldw]),1);
		work[2*j-3] = cblas_ddot(mloc,&(U[(j-2)*ldu]),1,&(W[j*ldw]),1);
		work[2*j-2] = cblas_ddot(mloc,&(W[(j-1)*ldw]),1,&(W[(j-1)*ldw]),1);
		work[2*j-1] = cblas_ddot(mloc,&(W[(j-1)*ldw]),1,&(W[j*ldw]),1);
		work[2*j] = cblas_ddot(mloc,&(W[(j-2)*mloc]),1,&(W[(j-2)*ldw]),1);
		MPI_Allreduce( MPI_IN_PLACE, &(work[0]), 2*j+1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		for( i=0;i<j-2;i++) C[i+(j-2)*ldc] = work[i];
		for( i=0;i<j-2;i++) H[i+(j-1)*ldh] = work[j-2+i];
		C[(j-2)+(j-2)*ldc] = work[2*j-4];
		H[(j-2)+(j-1)*ldh] = work[2*j-3]; 
		rho[(j-2)] = sqrt( work[2*j-2] );
		H[(j-1)+(j-1)*ldh] = work[2*j-1];
		H[(j-2)+(j-3)*ldh] = sqrt( work[2*j] );
		for( i=0;i<mloc;i++) U[i+(j-1)*ldu] = W[i+(j-1)*ldw] / rho[(j-2)];
		for( i=0;i<mloc;i++) W[i+j*ldw] = W[i+j*ldw] / rho[(j-2)];
		for( i=0;i<j-1;i++)  H[i+(j-1)*ldh] = H[i+(j-1)*ldh] / rho[(j-2)];
		H[(j-1)+(j-1)*ldh] = H[(j-1)+(j-1)*ldh] / ( rho[(j-2)] * rho[(j-2)] );
		cblas_dgemv(CblasColMajor, CblasNoTrans, mloc, j-2, -1.0e0,  Q, mloc, H+(j-1)*ldh, 1, 1.0e0, &( W[j*ldw]), 1);
		cblas_dgemv(CblasColMajor, CblasNoTrans, mloc, j-2, -1.0e0,  Q, mloc, C+(j-2)*ldc, 1, 1.0e0, W+(j-1)*ldw, 1);
		for( i=0;i<mloc;i++) W[i+j*ldw] = W[i+j*ldw] - U[i+(j-2)*ldu]*H[(j-2)+(j-1)*ldh] - U[i+(j-1)*ldu]*H[(j-1)+(j-1)*ldh];
		for( i=0;i<mloc;i++) W[i+(j-1)*ldw] = W[i+(j-1)*ldw] - U[i+(j-2)*ldu]*C[(j-2)+(j-2)*ldc];
		for( i=0;i<j-1;i++) H[i+(j-2)*ldh] += C[i+(j-2)*ldc];
		for( i=0;i<mloc;i++ ) Q[i+(j-2)*ldq] = W[i+(j-2)*ldw] / H[(j-2)+(j-3)*ldh];

	}


	return 0;

}
