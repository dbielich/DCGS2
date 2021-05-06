#include "../c-mpi.h"

int mpi_orth_cgs2_threesynch_append1(int mloc, int i, double *A, int lda, double *r, double *h, MPI_Comm mpi_comm){

	int j;

//
//	USUAL CGS2
//

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

//double normV;
//normV = cblas_ddot(mloc, &(A[i*mloc]),1,&(A[i*mloc]),1);  
//normV = sqrt( normV );
//for( j=0;j<mloc;j++ ) A[j+i*mloc] += ( ( (double) rand() ) / (double)(RAND_MAX) - 0.5e+00 ) * normV * 1e-12;

		cblas_dgemv(CblasColMajor, CblasTrans, mloc, i, 1.0e0,  A, mloc, &(A[i*mloc]), 1,
			0.0e0, &(h[0]), 1);
		MPI_Allreduce( MPI_IN_PLACE, &(h[0]), i, MPI_DOUBLE, MPI_SUM, mpi_comm);
		cblas_dgemv(CblasColMajor, CblasNoTrans, mloc, i, (-1.0e0),  A, mloc, &(h[0]), 1,
			1.0e0, &(A[i*mloc]), 1);

		for(j=0;j<i;j++) r[j] += h[j];

		// CGS3
//		cblas_dgemv(CblasColMajor, CblasTrans, mloc, i, 1.0e0,  A, mloc, &(A[i*mloc]), 1,
//			0.0e0, &(h[0]), 1);
//		MPI_Allreduce( MPI_IN_PLACE, &(h[0]), i, MPI_DOUBLE, MPI_SUM, mpi_comm);
//		cblas_dgemv(CblasColMajor, CblasNoTrans, mloc, i, (-1.0e0),  A, mloc, &(h[0]), 1,
//			1.0e0, &(A[i*mloc]), 1);

//		for(j=0;j<i;j++) r[j] += h[j];

	
		r[i] = cblas_ddot(mloc,&(A[i*mloc]),1,&(A[i*mloc]),1);
		MPI_Allreduce( MPI_IN_PLACE, &(r[i]), 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
		r[i] = sqrt(r[i]);

		cblas_dscal(mloc,1/r[i],&(A[i*mloc]),1);

	}

/*
//
//	DEFLATION OF USUAL CGS2
//
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

//double normV;
//normV = cblas_ddot(mloc, &(A[i*mloc]),1,&(A[i*mloc]),1);  
//normV = sqrt( normV );
//for( j=0;j<mloc;j++ ) A[j+i*mloc] += ( ( (double) rand() ) / (double)(RAND_MAX) - 0.5e+00 ) * normV * 1e-12;

		cblas_dgemv(CblasColMajor, CblasTrans, mloc, i, 1.0e0,  A, mloc, &(A[i*mloc]), 1,
			0.0e0, &(h[0]), 1);
		MPI_Allreduce( MPI_IN_PLACE, &(h[0]), i, MPI_DOUBLE, MPI_SUM, mpi_comm);
		cblas_dgemv(CblasColMajor, CblasNoTrans, mloc, i, (-1.0e0),  A, mloc, &(h[0]), 1,
			1.0e0, &(A[i*mloc]), 1);

		for(j=0;j<i;j++) r[j] += h[j];
	
		r[i] = cblas_ddot(mloc,&(A[i*mloc]),1,&(A[i*mloc]),1);
		MPI_Allreduce( MPI_IN_PLACE, &(r[i]), 1, MPI_DOUBLE, MPI_SUM, mpi_comm);

		r[i] = sqrt(r[i]);
		double alpha=0.0e00, beta=0.0e00;
		for(j=0;j<i;j++) alpha += r[j] * r[j];
		alpha = sqrt( alpha );

		beta = sqrt( alpha * alpha + r[i] * r[i] );

		if( beta * beta - alpha * alpha > 1e-15 * beta * beta ){
			cblas_dscal(mloc,1/r[i],&(A[i*mloc]),1);
		} else {

			r[i] = 0e00;
			for(j=0;j<mloc;j++) *(A+j+i*mloc) = ( ( (double) rand() ) / (double)(RAND_MAX) - 0.5e+00 );

			cblas_dgemv(CblasColMajor, CblasTrans, mloc, i, 1.0e0,  A, mloc, &(A[i*mloc]), 1,
				0.0e0, &(h[0]), 1);
			MPI_Allreduce( MPI_IN_PLACE, &(h[0]), i, MPI_DOUBLE, MPI_SUM, mpi_comm);
			cblas_dgemv(CblasColMajor, CblasNoTrans, mloc, i, (-1.0e0),  A, mloc, &(h[0]), 1,
				1.0e0, &(A[i*mloc]), 1);

			cblas_dgemv(CblasColMajor, CblasTrans, mloc, i, 1.0e0,  A, mloc, &(A[i*mloc]), 1,
				0.0e0, &(h[0]), 1);
			MPI_Allreduce( MPI_IN_PLACE, &(h[0]), i, MPI_DOUBLE, MPI_SUM, mpi_comm);
			cblas_dgemv(CblasColMajor, CblasNoTrans, mloc, i, (-1.0e0),  A, mloc, &(h[0]), 1,
				1.0e0, &(A[i*mloc]), 1);
	
			h[0] = cblas_ddot(mloc,&(A[i*mloc]),1,&(A[i*mloc]),1);
			MPI_Allreduce( MPI_IN_PLACE, &(h[0]), 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
			h[0] = sqrt( h[0] );
			cblas_dscal(mloc,1/h[0],&(A[i*mloc]),1);


		}

	}


*/

//
//	PYTHGOREAN TRICK ONE - BEFORE PROJECTION
//
/*
	double alpha, beta;

	if( i == 0){

		r[0] = cblas_ddot(mloc,A,1,A,1);
		MPI_Allreduce( MPI_IN_PLACE, &(r[0]), 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
		r[0] = sqrt(r[0]);
		cblas_dscal(mloc,1/r[0],A,1);

	} else {

		cblas_dgemv(CblasColMajor, CblasTrans, mloc, i, 1.0e0,  A, mloc, &(A[i*mloc]), 1, 0.0e0, &(r[0]), 1);
		beta = cblas_ddot(mloc,A+i*mloc,1,A+i*mloc,1);  

		MPI_Allreduce( MPI_IN_PLACE, &(beta), 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
		MPI_Allreduce( MPI_IN_PLACE, &(r[0]), i, MPI_DOUBLE, MPI_SUM, mpi_comm);

		cblas_dgemv(CblasColMajor, CblasNoTrans, mloc, i, (-1.0e0),  A, mloc, &(r[0]), 1, 1.0e0, &(A[i*mloc]), 1);

//double normV;
//normV = cblas_ddot(mloc, &(A[i*mloc]),1,&(A[i*mloc]),1);  
//normV = sqrt( normV );
//for( j=0;j<mloc;j++ ) A[j+i*mloc] += ( ( (double) rand() ) / (double)(RAND_MAX) - 0.5e+00 ) * normV * 1e-12;

		cblas_dgemv(CblasColMajor, CblasTrans, mloc, i, 1.0e0,  A, mloc, &(A[i*mloc]), 1, 0.0e0, &(h[0]), 1);

		MPI_Allreduce( MPI_IN_PLACE, &(h[0]), i, MPI_DOUBLE, MPI_SUM, mpi_comm);

		cblas_dgemv(CblasColMajor, CblasNoTrans, mloc, i, (-1.0e0),  A, mloc, &(h[0]), 1, 1.0e0, &(A[i*mloc]), 1);

		for(j=0;j<i;j++) r[j] += h[j];

//beta = beta * ( 1.0e00 + 1e-10 );

		alpha = 0.0e00;	for(j=0;j<i;j++) alpha += r[j] * r[j];
		r[i] = sqrt( beta - alpha );

//		alpha = 0.0e00;	for(j=0;j<i;j++) alpha += r[j] * r[j];
//		beta = sqrt( beta );
//		alpha = sqrt( alpha );

		r[i] = sqrt( (beta+alpha) * (beta-alpha) );

		cblas_dscal(mloc,1/r[i],&(A[i*mloc]),1);

	}
*/
/*
//
//	PYTHGOREAN TRICK ONE - BEFORE PROJECTION  ----   WITH DEFLATION
//

	double alpha, beta;

	if( i == 0){

		r[0] = cblas_ddot(mloc,A,1,A,1);
		MPI_Allreduce( MPI_IN_PLACE, &(r[0]), 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
		r[0] = sqrt(r[0]);
		cblas_dscal(mloc,1/r[0],A,1);

	} else {

		cblas_dgemv(CblasColMajor, CblasTrans, mloc, i, 1.0e0,  A, mloc, &(A[i*mloc]), 1, 0.0e0, &(r[0]), 1);

		beta = cblas_ddot(mloc,A+i*mloc,1,A+i*mloc,1);  

		MPI_Allreduce( MPI_IN_PLACE, &(beta), 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
		MPI_Allreduce( MPI_IN_PLACE, &(r[0]), i, MPI_DOUBLE, MPI_SUM, mpi_comm);

		cblas_dgemv(CblasColMajor, CblasNoTrans, mloc, i, (-1.0e0),  A, mloc, &(r[0]), 1, 1.0e0, &(A[i*mloc]), 1);

//double normV;
//normV = cblas_ddot(mloc, &(A[i*mloc]),1,&(A[i*mloc]),1);  
//normV = sqrt( normV );
//for( j=0;j<mloc;j++ ) A[j+i*mloc] += ( ( (double) rand() ) / (double)(RAND_MAX) - 0.5e+00 ) * normV * 1e-12;

		cblas_dgemv(CblasColMajor, CblasTrans, mloc, i, 1.0e0,  A, mloc, &(A[i*mloc]), 1, 0.0e0, &(h[0]), 1);

		MPI_Allreduce( MPI_IN_PLACE, &(h[0]), i, MPI_DOUBLE, MPI_SUM, mpi_comm);

		cblas_dgemv(CblasColMajor, CblasNoTrans, mloc, i, (-1.0e0),  A, mloc, &(h[0]), 1, 1.0e0, &(A[i*mloc]), 1);

		for(j=0;j<i;j++) r[j] += h[j];

		alpha = 0.0e00; for(j=0;j<i;j++) alpha += r[j] * r[j];

beta = beta * ( 1.0e00 + 1e-10 );

		if( beta - alpha > 1e-15 * beta  ){
			r[i] = sqrt( beta - alpha );
			cblas_dscal(mloc,1/r[i],&(A[i*mloc]),1);

		} else {

			r[i] = 0e00;
			for(j=0;j<mloc;j++) *(A+j+i*mloc) = ( ( (double) rand() ) / (double)(RAND_MAX) - 0.5e+00 );

			cblas_dgemv(CblasColMajor, CblasTrans, mloc, i, 1.0e0,  A, mloc, &(A[i*mloc]), 1,
				0.0e0, &(h[0]), 1);
			MPI_Allreduce( MPI_IN_PLACE, &(h[0]), i, MPI_DOUBLE, MPI_SUM, mpi_comm);
			cblas_dgemv(CblasColMajor, CblasNoTrans, mloc, i, (-1.0e0),  A, mloc, &(h[0]), 1,
				1.0e0, &(A[i*mloc]), 1);

			cblas_dgemv(CblasColMajor, CblasTrans, mloc, i, 1.0e0,  A, mloc, &(A[i*mloc]), 1,
				0.0e0, &(h[0]), 1);
			MPI_Allreduce( MPI_IN_PLACE, &(h[0]), i, MPI_DOUBLE, MPI_SUM, mpi_comm);
			cblas_dgemv(CblasColMajor, CblasNoTrans, mloc, i, (-1.0e0),  A, mloc, &(h[0]), 1,
				1.0e0, &(A[i*mloc]), 1);
	
			h[0] = cblas_ddot(mloc,&(A[i*mloc]),1,&(A[i*mloc]),1);
			MPI_Allreduce( MPI_IN_PLACE, &(h[0]), 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
			h[0] = sqrt( h[0] );
			cblas_dscal(mloc,1/h[0],&(A[i*mloc]),1);
		}

	}
*/

//
//	PYTHGOREAN TRICK TWO - AFTER PROJECTION  
//
/*
	double alpha, beta;

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

//double normV;
//normV = cblas_ddot(mloc, &(A[i*mloc]),1,&(A[i*mloc]),1);  
//normV = sqrt( normV );
//for( j=0;j<mloc;j++ ) A[j+i*mloc] += ( ( (double) rand() ) / (double)(RAND_MAX) - 0.5e+00 ) * normV * 1e-12;

		cblas_dgemv(CblasColMajor, CblasTrans, mloc, i, 1.0e0,  A, mloc, &(A[i*mloc]), 1,
			0.0e0, &(h[0]), 1);

		beta = cblas_ddot(mloc,A+i*mloc,1,A+i*mloc,1);  

		MPI_Allreduce( MPI_IN_PLACE, &(beta), 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
		MPI_Allreduce( MPI_IN_PLACE, &(h[0]), i, MPI_DOUBLE, MPI_SUM, mpi_comm);

		cblas_dgemv(CblasColMajor, CblasNoTrans, mloc, i, (-1.0e0),  A, mloc, &(h[0]), 1,
			1.0e0, &(A[i*mloc]), 1);

		for(j=0;j<i;j++) r[j] += h[j];

//beta = beta * ( 1.0e00 + 1e-10 );

//		alpha = 0.0e00;
//		for(j=0;j<i;j++) alpha += h[j] * h[j];
//		r[i] = sqrt( beta - alpha );

		alpha = 0.0e00;
		for(j=0;j<i;j++) alpha += h[j] * h[j];
		beta = sqrt( beta );
		alpha = sqrt( alpha );
		r[i] = sqrt( ( beta + alpha ) * ( beta - alpha ) );

		cblas_dscal(mloc,1/r[i],&(A[i*mloc]),1);

	}
*/
/*
//
//	PYTHGOREAN TRICK TWO - AFTER PROJECTION  ----   WITH DEFLATION
//

	double alpha, beta;

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

//double normV;
//normV = cblas_ddot(mloc, &(A[i*mloc]),1,&(A[i*mloc]),1);  
//normV = sqrt( normV );
//for( j=0;j<mloc;j++ ) A[j+i*mloc] += ( ( (double) rand() ) / (double)(RAND_MAX) - 0.5e+00 ) * normV * 1e-12;

		cblas_dgemv(CblasColMajor, CblasTrans, mloc, i, 1.0e0,  A, mloc, &(A[i*mloc]), 1,
			0.0e0, &(h[0]), 1);

		beta = cblas_ddot(mloc,A+i*mloc,1,A+i*mloc,1);  

		MPI_Allreduce( MPI_IN_PLACE, &(beta), 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
		MPI_Allreduce( MPI_IN_PLACE, &(h[0]), i, MPI_DOUBLE, MPI_SUM, mpi_comm);

		cblas_dgemv(CblasColMajor, CblasNoTrans, mloc, i, (-1.0e0),  A, mloc, &(h[0]), 1,
			1.0e0, &(A[i*mloc]), 1);

		for(j=0;j<i;j++) r[j] += h[j];

beta = beta * ( 1.0e00 + 1e-10 );

		alpha=0.0e00; for(j=0;j<i;j++) alpha += h[j] * h[j]; // NOTE THE DIFFERENCE, HERE WE ONLY LOOK AT h

		if( beta - alpha > 1e-15 * beta  ){
			r[i] = sqrt( beta - alpha );
			cblas_dscal(mloc,1/r[i],&(A[i*mloc]),1);
		} else {

			r[i] = 0e00;
			for(j=0;j<mloc;j++) *(A+j+i*mloc) = ( ( (double) rand() ) / (double)(RAND_MAX) - 0.5e+00 );

			cblas_dgemv(CblasColMajor, CblasTrans, mloc, i, 1.0e0,  A, mloc, &(A[i*mloc]), 1,
				0.0e0, &(h[0]), 1);
			MPI_Allreduce( MPI_IN_PLACE, &(h[0]), i, MPI_DOUBLE, MPI_SUM, mpi_comm);
			cblas_dgemv(CblasColMajor, CblasNoTrans, mloc, i, (-1.0e0),  A, mloc, &(h[0]), 1,
				1.0e0, &(A[i*mloc]), 1);

			cblas_dgemv(CblasColMajor, CblasTrans, mloc, i, 1.0e0,  A, mloc, &(A[i*mloc]), 1,
				0.0e0, &(h[0]), 1);
			MPI_Allreduce( MPI_IN_PLACE, &(h[0]), i, MPI_DOUBLE, MPI_SUM, mpi_comm);
			cblas_dgemv(CblasColMajor, CblasNoTrans, mloc, i, (-1.0e0),  A, mloc, &(h[0]), 1,
				1.0e0, &(A[i*mloc]), 1);
	
			h[0] = cblas_ddot(mloc,&(A[i*mloc]),1,&(A[i*mloc]),1);
			MPI_Allreduce( MPI_IN_PLACE, &(h[0]), 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
			h[0] = sqrt( h[0] );
			cblas_dscal(mloc,1/h[0],&(A[i*mloc]),1);
		}
	}
*/
	return 0;

}
