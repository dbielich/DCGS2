#include "../c-mpi.h"

int mpi_orth_dcgs2_onesynch_append1_qr( int mloc, int j, double *Q, int ldq, double *R, int ldr, double *work, MPI_Comm mpi_comm){

	int i;
	double tmp;

	if ( j == 1 ){

		cblas_dgemv(CblasColMajor, CblasTrans, mloc, 1, 1.0e0,  Q, ldq, Q, 1, 0.0e0, &(work[0]), 1);
		cblas_dgemv(CblasColMajor, CblasTrans, mloc, 1, 1.0e0,  Q, ldq, Q+ldq, 1, 0.0e0, &(work[1]), 1);

		MPI_Allreduce( MPI_IN_PLACE, &(work[0]), 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		R[0] = sqrt( work[0] );
		cblas_dscal(ldq,1/R[0],Q,1);
		R[ldr] = work[1] / R[0];

		for( i=0;i<ldq;i++ ) Q[i+ldq] = Q[i+ldq] - Q[i]*R[ldr]; 

	}

	if ( j >= 2 ){

                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, j, 2, mloc, 1.0e0, Q, ldq, &(Q[(j-1)*ldq]), ldq, 0.0e0, &(work[0]), j);
		MPI_Allreduce( MPI_IN_PLACE, &(work[0]), 2*j, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	        for(i=0;i<j;i++) R[i+j*ldr] = work[i+j]; 
		R[j-1+(j-1)*ldr] = work[j-1];

		tmp = 0.0e+00; for(i=0;i<j-1;i++) tmp += R[i+j*ldr] * work[i];   
	        R[j-1+j*ldr] = R[j-1+j*ldr] - tmp;

	        tmp = 0.0e+00; for(i=0;i<j-1;i++) tmp += work[i] * work[i];
	        R[j-1+(j-1)*ldr] = sqrt( R[j-1+(j-1)*ldr] - tmp );

		for(i=0;i<j-1;i++) R[i+(j-1)*ldr] = R[i+(j-1)*ldr] + work[i]; 
	        R[j-1+j*ldr] = R[j-1+j*ldr] / R[j-1+(j-1)*ldr];

		for(i=0;i<j-1;i++) work[i+(j-1)] = R[i+j*ldr];
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, mloc, 2, j-1, -1.0e0, Q, ldq, &(work[0]), j-1, 1.0e0, &(Q[(j-1)*ldq]), ldq);

		cblas_dscal(ldq,(1.0e+00/R[j-1+(j-1)*ldr]),Q+(j-1)*ldq,1);
		for( i=0;i<ldq;i++ ) Q[i+j*ldq] = Q[i+j*ldq] - Q[i+(j-1)*ldq]*R[j-1+j*ldr]; 

	}

	return 0;

}
