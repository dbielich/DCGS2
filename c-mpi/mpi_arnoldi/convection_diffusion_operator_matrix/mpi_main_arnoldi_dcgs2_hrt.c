#include "../../c-mpi.h"

int main(int argc, char ** argv) {

	double *W, *U, *C, *Q, *H, *b, *rho, *work, beta;
	double *y, *v, BEETA, L, h;
	double norm_orth, norm_repres;
	double elapse, tol;
	long int m;
	int n, k, i, j, testing;
	int my_rank, pool_size, seed, mloc;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &pool_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    	n         = 2*pool_size;
	k         = 11;
	testing   =  0;
	tol       = 1.00e-7;
	BEETA     = 0.5;

	for(i = 1; i < argc; i++){
		if( strcmp( *(argv + i), "-testing") == 0) {
			testing = atoi( *(argv + i + 1) );
			i++;
		}
		if( strcmp( *(argv + i), "-n") == 0) {
			n = atoi( *(argv + i + 1) );
			i++;
		}
		if( strcmp( *(argv + i), "-k") == 0) {
			k = atoi( *(argv + i + 1) );
			i++;
		}
		if( strcmp( *(argv + i), "-tol") == 0) {
			tol = atof( *(argv + i + 1) );
			i++;
		}
		if( strcmp( *(argv + i), "-b") == 0) {
			BEETA = atof( *(argv + i + 1) );
			i++;
		}
	}
	m = ( n * n );
        L = n+1;
 	h = L / (n+1);

	seed = my_rank*m*m; srand(seed);
	mloc = m / pool_size; if ( my_rank < m - pool_size*mloc ) mloc++;

	int stop=0; if( mloc < k ){ stop = 1; }
	MPI_Allreduce( MPI_IN_PLACE, &stop, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	if ( stop > 0 ){ printf("\n\n We need k <= mloc\n\n"); MPI_Finalize(); return 0; }
	if ( pool_size >= n ){ printf("\n\n We need pool_size < n\n\n"); MPI_Finalize(); return 0; }

	y = (double *) malloc( mloc * sizeof(double));
	v = (double *) malloc( mloc * sizeof(double));

	work = (double *) malloc( (5*k) * sizeof(double));

	rho = (double *) malloc( (k+1) * sizeof(double));

	b = (double *) malloc( mloc * sizeof(double));
	H = (double *) malloc( (k+1) * k * sizeof(double));
	C = (double *) malloc( (k+1) * k * sizeof(double));
	Q = (double *) malloc( mloc * (k+1) * sizeof(double));
	U = (double *) malloc( mloc * (2*k+1) * sizeof(double));
	W = (double *) malloc( mloc * (2*k+1) * sizeof(double));

	double *x_notlocal;
	x_notlocal = (double *) malloc( 2 * n * sizeof(double));	

 	for(i = 0; i < mloc; i++){
		*(b + i) = ( (double) 1 );
	}

	MPI_Barrier( MPI_COMM_WORLD );
	elapse = -MPI_Wtime();
//
//	Initialization - j == 0
//
	cblas_dcopy( mloc, b, 1, W, 1);
	beta = cblas_ddot(mloc,W,1,W,1);
	MPI_Allreduce( MPI_IN_PLACE, &(beta), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	beta = sqrt(beta);
	H[0] = beta;

	for( j = 1; j < k+1 ; j++ ){

		mpi_matvec_manteuffel_M( MPI_COMM_WORLD, m, W+(j-1)*mloc, y );
		mpi_matvec_manteuffel_M_nblock_eo( MPI_COMM_WORLD, m, W+(j-1)*mloc, y, x_notlocal );
		mpi_matvec_manteuffel_N( MPI_COMM_WORLD, m, W+(j-1)*mloc, v );
		mpi_matvec_manteuffel_N_nblock_eo( MPI_COMM_WORLD, m, W+(j-1)*mloc, v, x_notlocal );

		for( i=0;i<mloc;i++ ){ W[i+j*mloc] = ( 1 / ( h * h ) ) * y[i] + ( BEETA / ( 2 * h ) ) * v[i]; }

//		mpi_orth_dcgs2_hrt_onesynch_append1_arnoldi( mloc, j, W, mloc, U, mloc, Q, mloc, H, k+1, C, k+1, rho, work, MPI_COMM_WORLD);	


		if ( j == 1 ){


			H[0] = cblas_ddot(mloc,&(W[(0)*mloc]),1,&(W[1*mloc]),1);
			MPI_Allreduce( MPI_IN_PLACE, &(H[0]), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			H[0] = H[0] / ( beta * beta );
			for( i=0;i<mloc;i++ ) U[i+(0)*mloc] = W[i+(0)*mloc] / beta;
			cblas_dscal(mloc,1/beta,&(W[1*mloc]),1);
			cblas_daxpy( mloc, -H[0], &(U[0]), 1, &(W[1*mloc]), 1 );

		}

		if ( j == 2 ){


			work[0] = cblas_ddot(mloc,&(W[(1)*mloc]),1,&(W[1*mloc]),1);
			work[1] = cblas_ddot(mloc,&(U[(0)*mloc]),1,&(W[2*mloc]),1);
			work[2] = cblas_ddot(mloc,&(W[1*mloc]),1,&(W[2*mloc]),1);
			work[3] = 0.0e00; for( i=0;i<mloc;i++ ) work[3] += U[i+(0)*mloc] * W[i+1*mloc];
			MPI_Allreduce( MPI_IN_PLACE, work, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			C[0] = work[3];
			rho[0] = sqrt( work[0] );
			H[0+1*(k+1)] = work[1];
			H[1+1*(k+1)] = work[2];
			for( i=0;i<mloc;i++ ) U[i+1*mloc] = W[i+1*mloc] / rho[0];
			for( i=0;i<mloc;i++ ) W[i+2*mloc] = W[i+2*mloc] / rho[0];
			for( i=0;i<1;i++ )  H[i+1*(k+1)] = H[i+1*(k+1)] / rho[0];
			H[1+1*(k+1)] = H[1+1*(k+1)] / ( rho[0] * rho[0] );
			for( i=0;i<mloc;i++ ) W[i+2*mloc] = W[i+2*mloc] - U[i+(0)*mloc]*H[(0)+1*(k+1)] - U[i+(1)*mloc]*H[(1)+1*(k+1)];
			for( i=0;i<mloc;i++ ) W[i+1*mloc] = W[i+1*mloc] - U[i+(0)*mloc]*C[0];
			H[0] += C[0];
			for( i=0;i<mloc;i++ ) Q[i+(0)*mloc] = U[i+(0)*mloc];

		}

		if ( j >= 3 ){


			cblas_dgemv(CblasColMajor, CblasTrans, mloc, (j-2), 1.0e0,  Q, mloc, W+(j-1)*mloc, 1, 0.0e0, &(work[0]), 1);
			cblas_dgemv(CblasColMajor, CblasTrans, mloc, (j-2), 1.0e0,  Q, mloc, &(W[j*mloc]), 1, 0.0e0, &(work[j-2]), 1);
			work[2*j-4] = cblas_ddot(mloc,&(U[(j-2)*mloc]),1,&(W[(j-1)*mloc]),1);
			work[2*j-3] = cblas_ddot(mloc,&(U[(j-2)*mloc]),1,&(W[j*mloc]),1);
			work[2*j-2] = cblas_ddot(mloc,&(W[(j-1)*mloc]),1,&(W[(j-1)*mloc]),1);
			work[2*j-1] = cblas_ddot(mloc,&(W[(j-1)*mloc]),1,&(W[j*mloc]),1);
			work[2*j] = cblas_ddot(mloc,&(W[(j-2)*mloc]),1,&(W[(j-2)*mloc]),1);
			MPI_Allreduce( MPI_IN_PLACE, &(work[0]), 2*j+1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			for( i=0;i<j-2;i++) C[i+(j-2)*(k+1)] = work[i];
			for( i=0;i<j-2;i++) H[i+(j-1)*(k+1)] = work[j-2+i];
			C[(j-2)+(j-2)*(k+1)] = work[2*j-4];
			H[(j-2)+(j-1)*(k+1)] = work[2*j-3]; 
			rho[(j-2)] = sqrt( work[2*j-2] );
			H[(j-1)+(j-1)*(k+1)] = work[2*j-1];
			H[(j-2)+(j-3)*(k+1)] = sqrt( work[2*j] );
			for( i=0;i<mloc;i++) U[i+(j-1)*mloc] = W[i+(j-1)*mloc] / rho[(j-2)];
			for( i=0;i<mloc;i++) W[i+j*mloc] = W[i+j*mloc] / rho[(j-2)];
			for( i=0;i<j-1;i++)  H[i+(j-1)*(k+1)] = H[i+(j-1)*(k+1)] / rho[(j-2)];
			H[(j-1)+(j-1)*(k+1)] = H[(j-1)+(j-1)*(k+1)] / ( rho[(j-2)] * rho[(j-2)] );
			cblas_dgemv(CblasColMajor, CblasNoTrans, mloc, j-2, -1.0e0,  Q, mloc, H+(j-1)*(k+1), 1, 1.0e0, &( W[j*mloc]), 1);
			cblas_dgemv(CblasColMajor, CblasNoTrans, mloc, j-2, -1.0e0,  Q, mloc, C+(j-2)*(k+1), 1, 1.0e0, W+(j-1)*mloc, 1);
			for( i=0;i<mloc;i++) W[i+j*mloc] = W[i+j*mloc] - U[i+(j-2)*mloc]*H[(j-2)+(j-1)*(k+1)] - U[i+(j-1)*mloc]*H[(j-1)+(j-1)*(k+1)];
			for( i=0;i<mloc;i++) W[i+(j-1)*mloc] = W[i+(j-1)*mloc] - U[i+(j-2)*mloc]*C[(j-2)+(j-2)*(k+1)];
			for( i=0;i<j-1;i++) H[i+(j-2)*(k+1)] += C[i+(j-2)*(k+1)];
			for( i=0;i<mloc;i++ ) Q[i+(j-2)*mloc] = W[i+(j-2)*mloc] / H[(j-2)+(j-3)*(k+1)];

		}



		if ( j == k ){

			H[(k-1)+(k-2)*(k+1)] = cblas_ddot(mloc,&(W[(k-1)*mloc]),1,&(W[(k-1)*mloc]),1);
			cblas_dgemv(CblasColMajor, CblasTrans, mloc, k-1, 1.0e0,  Q, mloc, &(W[k*mloc]), 1, 0.0e0, &(C[(k-1)*(k+1)]), 1);
			cblas_dgemv(CblasColMajor, CblasNoTrans, mloc, k-1, -1.0e0,  Q, mloc, &(C[(k-1)*(k+1)]), 1, 1.0e0, &(W[k*mloc]), 1);
			MPI_Allreduce( MPI_IN_PLACE, &(H[(k-1)+(k-2)*(k+1)]), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			H[(k-1)+(k-2)*(k+1)] = sqrt( H[(k-1)+(k-2)*(k+1)] );
			for( i=0;i<mloc;i++) Q[i+(k-1)*mloc] = W[i+(k-1)*mloc] / H[(k-1)+(k-2)*(k+1)];
			for( i=0;i<k-1;i++) H[i+(k-1)*(k+1)] += C[i+(k-1)*(k+1)];
			H[k+(k-1)*(k+1)] = cblas_ddot(mloc,&(W[i+k*mloc]),1,&(W[i+k*mloc]),1);
			MPI_Allreduce( MPI_IN_PLACE, &(H[k+(k-1)*(k+1)]), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce( MPI_IN_PLACE, &(C[(k-1)*(k+1)]), k-1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			H[k+(k-1)*(k+1)] = sqrt( H[k+(k-1)*(k+1)] );
			for( i=0;i<mloc;i++) Q[i+k*mloc] = W[i+k*mloc] / H[k+(k-1)*(k+1)];
			
		}


	}

//	mpi_orth_dcgs2_hrt_onesynch_cleanup_append1_arnoldi( mloc, k, W, mloc, U, mloc, Q, mloc, H, k+1, C, k+1, rho, work, MPI_COMM_WORLD);	

	MPI_Barrier( MPI_COMM_WORLD );
	elapse += MPI_Wtime();

	free( W );	
	free( U );
	free( C );
	free( rho );

	if ( testing ){

		mpi_check_qq_orth( MPI_COMM_WORLD, &norm_orth, mloc, k, Q, mloc  );
		mpi_check_manteuffel_arnoldi_repres( MPI_COMM_WORLD, &norm_repres, m, mloc, k-1, h, BEETA, b, Q, mloc, beta, H, k+1 );
		if( my_rank == 0 ){ printf("%3d %8ld %6d %6.2f  %6.2e %6.6e  %8.4e %8.4e", 
					pool_size, m, k, BEETA, tol, elapse, norm_repres, norm_orth );
		}

		if( my_rank == 0 ) printf("\n");


	}
	if ( !testing ){
		if( my_rank == 0 ){ printf("%3d %8ld %6d %6.2f  %6.2e %6.6e", pool_size, m, k, BEETA, tol, elapse ); }
		if( my_rank == 0 ) printf("\n");
	}
	free( y );
	free( v );
	free( x_notlocal );

	free( Q );
	free( H );
	free( b );



        MPI_Finalize();
	return 0;

}
