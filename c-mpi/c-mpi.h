#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include "mpi.h"

#if !defined(USE_MKL)
#include "cblas.h"
#include "lapacke.h"
#endif



extern int mpi_matvec( MPI_Comm mpi_comm, long int m, double *x, double *y );
extern int mpi_matvec_manteuffel_M( MPI_Comm mpi_comm, long int m, double *x, double *y );
extern int mpi_matvec_manteuffel_N( MPI_Comm mpi_comm, long int m, double *x, double *y );
extern int mpi_matvec_manteuffel_N_nblock_eo( MPI_Comm mpi_comm, long int m, double *x, double *y, double *x_notlocal );
extern int mpi_matvec_manteuffel_M_nblock_eo( MPI_Comm mpi_comm, long int m, double *x, double *y, double *x_notlocal );

extern int mpi_check_qr_repres( MPI_Comm mpi_comm, double *norm_repres, int m, int n, double *A, int lda, double *Q, int ldq, double *R, int ldr );
extern int mpi_check_qq_orth  ( MPI_Comm mpi_comm, double *norm_orth_1, int mloc, int n, double *Q, int ldq  );
extern int mpi_check_arnoldi_repres ( MPI_Comm mpi_comm, double *norm_repres, long int m, int mloc, int k, double *b, double *Q, int ldq, double beta, double *H, int ldh );
extern int mpi_check_manteuffel_arnoldi_repres( MPI_Comm mpi_comm, double *norm_repres, long int m, int mloc, int k, double h, double BEETA, double *b, double *Q, int ldq, double beta, double *H, int ldh );


extern int mpi_orth_hh_lvl1_manysynch_append1( MPI_Comm mpi_comm, int mloc, int i, double *A, int lda, double *tau, double *q, double *r );
extern int mpi_orth_hh_lvl2_fivesynch_append1( MPI_Comm mpi_comm, int mloc, int j, double *A, int lda, double *T, int ldt, double *q, double *r );
extern int mpi_orth_mgs_lvl1_manysynch_append1(int mloc, int i, double *A, int lda, double *r, MPI_Comm mpi_comm);
extern int mpi_orth_mgs_lvl2_icwy_threesynch_append1(int mloc, int i, double *A, int lda, double *T, int ldt, double *r, MPI_Comm mpi_comm);
extern int mpi_orth_mgs_lvl2_cwy_threesynch_append1(int mloc, int i, double *A, int lda, double *T, int ldt, double *r, MPI_Comm mpi_comm);
extern int mpi_orth_cgs_twosynch_append1(int mloc, int i, double *A, int lda, double *r, MPI_Comm mpi_comm);
extern int mpi_orth_cgs2_threesynch_append1(int mloc, int i, double *A, int lda, double *r, double *h, MPI_Comm mpi_comm);
extern int mpi_orth_dcgs2_onesynch_append1_arnoldi( int mloc, int j, double *Q, int ldq, double *H, int ldh, double *work, MPI_Comm mpi_comm);
extern int mpi_orth_dcgs2_onesynch_cleanup_append1_arnoldi( int mloc, int j, double *Q, int ldq, double *H, int ldh, double *work, MPI_Comm mpi_comm);
extern int mpi_orth_dcgs2_onesynch_append1_qr( int mloc, int j, double *Q, int ldq, double *R, int ldr, double *work, MPI_Comm mpi_comm);
extern int mpi_orth_dcgs2_onesynch_cleanup_append1_qr( int mloc, int j, double *Q, int ldq, double *R, int ldr, double *work, MPI_Comm mpi_comm);
extern int mpi_orth_dcgs2_hrt_onesynch_append1_arnoldi( int mloc, int j, double *W, int ldw, double *U, int ldu, double *Q, int ldq, double *H, int ldh, double *C, int ldc, double *rho, double *work, MPI_Comm mpi_comm);
extern int mpi_orth_dcgs2_hrt_onesynch_cleanup_append1_arnoldi( int mloc, int j, double *W, int ldw, double *U, int ldu, double *Q, int ldq, double *H, int ldh, double *C, int ldc, double *rho, double *work, MPI_Comm mpi_comm);

