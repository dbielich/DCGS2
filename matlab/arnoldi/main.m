%
   close all
%
   clear
%
   oldpath = path;
   path('../orth/',oldpath)
   oldpath = path;
   path('../matrix_market/',oldpath)
%
   m = 500;
   k = 50;
%
   b = ones(m,1);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   a = (m/2);
%   c = m-a;
%   d = 1000000000;
%
%   u = linspace( 0, 1, m-1 );
%   u = zeros(m-1,1);
%   
%   v = linspace( 2, 4, m-1 );
%   v = flip(v);
%   v = zeros(m-1,1);
%   
%   s = linspace( -c, a, m );
%   s = linspace( 1, d*m-1, m );
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   u = -1*ones(m-1,1);
%   v = -2*ones(m-1,1);
%  u = zeros(m-1,1);
%  v = zeros(m-1,1);
%  s = 10.^[ (1:m) ];
%   s = 10*ones(m,1);
%	   
%   U = diag(u,1);
%   V = diag(v,-1);
%   S = diag( s );
%   A = S + V + U;
%   clear U S V;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
  m = 100; 
  A = diag( linspace(1,m,m) );
  A(1,1) = 1e-8;
  b = randn(m,1);
  k = 80; 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
%
%   [A] = mmread('thermal1.mtx');
%   m = size(A,1);
%   b = ones(m,1);
%   k = 120;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
   A = sparse(A);
%
  [ Q, H, beta ] = arnoldi__orth_cgs     ( A, b, k ); [ repres_cgs,      orth_cgs,      cond_cgs, ~ ]      = arnoldi__orth_stabilitymetric( A, b, Q, H, beta ); clear Q H beta;
  [ Q, H, beta ] = arnoldi__orth_cgs2    ( A, b, k ); [ repres_cgs2,     orth_cgs2,     cond_cgs2, ~ ]     = arnoldi__orth_stabilitymetric( A, b, Q, H, beta ); clear Q H beta;
  [ Q, H, beta ] = arnoldi__orth_dcgs2   ( A, b, k ); [ repres_dcgs2,    orth_dcgs2,    cond_dcgs2, ~ ]    = arnoldi__orth_stabilitymetric( A, b, Q, H, beta ); clear Q H beta;
  [ Q, H, beta ] = arnoldi__orth_hh_lvl1 ( A, b, k ); [ repres_hh_lvl1,  orth_hh_lvl1,  cond_hh_lvl1, ~ ]  = arnoldi__orth_stabilitymetric( A, b, Q, H, beta ); clear Q H beta;
  [ Q, H, beta ] = arnoldi__orth_hh_lvl2 ( A, b, k ); [ repres_hh_lvl2,  orth_hh_lvl2,  cond_hh_lvl2, ~ ]  = arnoldi__orth_stabilitymetric( A, b, Q, H, beta ); clear Q H beta;
  [ Q, H, beta ] = arnoldi__orth_mgs_lvl1( A, b, k ); [ repres_mgs_lvl1, orth_mgs_lvl1, cond_mgs_lvl1, ~ ] = arnoldi__orth_stabilitymetric( A, b, Q, H, beta ); clear Q H beta;
  [ Q, H, beta ] = arnoldi__orth_mgs_lvl2( A, b, k ); [ repres_mgs_lvl2, orth_mgs_lvl2, cond_mgs_lvl2, ~ ] = arnoldi__orth_stabilitymetric( A, b, Q, H, beta ); clear Q H beta;
%
   x = [1:k];
%
   figure
   ax = axes;
   semilogy(x,orth_cgs,'-','LineWidth', 2, 'color', 'm','HandleVisibility','off'); hold on;
   semilogy(x(1:5:k),orth_cgs(1:5:k),'*-','LineWidth', 2, 'color', 'm'); hold on;
   semilogy(x,orth_cgs2,'-','LineWidth', 2, 'color', 'r','HandleVisibility','off'); hold on;
   semilogy(x(1:5:k),orth_cgs2(1:5:k),'*-','LineWidth', 2, 'color', 'r'); hold on;
   semilogy(x,orth_dcgs2,'-','LineWidth', 2, 'color', 'r','HandleVisibility','off'); hold on;
   semilogy(x(1:5:k),orth_dcgs2(1:5:k),'o-','LineWidth', 2, 'color', 'r'); hold on;
   semilogy(x,orth_hh_lvl1,'-','LineWidth', 2, 'color', 'b','HandleVisibility','off'); hold on;
   semilogy(x(1:5:k),orth_hh_lvl1(1:5:k),'*-','LineWidth', 2, 'color', 'b'); hold on;
   semilogy(x,orth_hh_lvl2,'-','LineWidth', 2, 'color', 'b','HandleVisibility','off'); hold on;
   semilogy(x(1:5:k),orth_hh_lvl2(1:5:k),'+-','LineWidth', 2, 'color', 'b'); hold on;
   semilogy(x,orth_mgs_lvl1,'-','LineWidth', 2, 'color', 'k','HandleVisibility','off'); hold on;
   semilogy(x(1:5:k),orth_mgs_lvl1(1:5:k),'*-','LineWidth', 2, 'color', 'k'); hold on;
   semilogy(x,orth_mgs_lvl2,'-','LineWidth', 2, 'color', 'k','HandleVisibility','off'); hold on;
   semilogy(x(1:5:k),orth_mgs_lvl2(1:5:k),'+-','LineWidth', 2, 'color', 'k'); hold on;
   grid on;
   axis([ min(x) max(x) 1e-16 1e2 ]);
   legend('Classical Gram-Schmidt', 'Classical Gram-Schmidt with Reorthogonalization', 'Delayed Classical Gram-Schmidt with Reorthogonalization', 'Householder - Level 1', 'Householder - Level 2', 'Modified Gram Schmidt - Level 1', 'Modified Gram Schmidt - Level 2','Location','northwest');
   title({'Orthogonality Comparison, Arnoldi expansion','m = 500, k = 200'});
   xlabel('Arnoldi Expansion')
   ylabel('Level of Orthogonality, || I - Q^T Q ||_f_r_o')
%
   figure
   ax = axes;
   semilogy(x,repres_cgs,'-','LineWidth', 2, 'color', 'm','HandleVisibility','off'); hold on;
   semilogy(x(1:5:k),repres_cgs(1:5:k),'*-','LineWidth', 2, 'color', 'm'); hold on;
   semilogy(x,repres_cgs2,'-','LineWidth', 2, 'color', 'r','HandleVisibility','off'); hold on;
   semilogy(x(1:5:k),repres_cgs2(1:5:k),'*-','LineWidth', 2, 'color', 'r'); hold on;
   semilogy(x,repres_dcgs2,'-','LineWidth', 2, 'color', 'r','HandleVisibility','off'); hold on;
   semilogy(x(1:5:k),repres_dcgs2(1:5:k),'o-','LineWidth', 2, 'color', 'r'); hold on;
   semilogy(x,repres_hh_lvl1,'-','LineWidth', 2, 'color', 'b','HandleVisibility','off'); hold on;
   semilogy(x(1:5:k),repres_hh_lvl1(1:5:k),'*-','LineWidth', 2, 'color', 'b'); hold on;
   semilogy(x,repres_hh_lvl2,'-','LineWidth', 2, 'color', 'b','HandleVisibility','off'); hold on;
   semilogy(x(1:5:k),repres_hh_lvl2(1:5:k),'+-','LineWidth', 2, 'color', 'b'); hold on;
   semilogy(x,repres_mgs_lvl1,'-','LineWidth', 2, 'color', 'k','HandleVisibility','off'); hold on;
   semilogy(x(1:5:k),repres_mgs_lvl1(1:5:k),'*-','LineWidth', 2, 'color', 'k'); hold on;
   semilogy(x,repres_mgs_lvl2,'-','LineWidth', 2, 'color', 'k','HandleVisibility','off'); hold on;
   semilogy(x(1:5:k),repres_mgs_lvl2(1:5:k),'+-','LineWidth', 2, 'color', 'k'); hold on;
   grid on;
   axis([ min(x) max(x) 1e-19 1e2 ]);
   legend('Classical Gram-Schmidt', 'Classical Gram-Schmidt with Reorthogonalization', 'Delayed Classical Gram-Schmidt with Reorthogonalization', 'Householder - Level 1', 'Householder - Level 2', 'Modified Gram Schmidt - Level 1', 'Modified Gram Schmidt - Level 2','Location','northwest');
   title({'Representativity Comparison, Arnoldi expansion','m = 500, k = 200'});
   xlabel('Arnoldi Expansion')
   ylabel('Level of Representativity, || A Q - Q H ||_f_r_o / || A ||_f_r_o')
%
   figure
   ax = axes;
   semilogy(x,cond_cgs,'-*','LineWidth', 2, 'color', 'm'); hold on;
   semilogy(x,cond_cgs2,'-*','LineWidth', 2, 'color', 'r'); hold on;
   semilogy(x,cond_dcgs2,'-o','LineWidth', 2, 'color', 'r'); hold on;
   semilogy(x,cond_hh_lvl1,'-*','LineWidth', 2, 'color', 'b'); hold on;
   semilogy(x,cond_hh_lvl2,'-+','LineWidth', 2, 'color', 'b'); hold on;
   semilogy(x,cond_mgs_lvl1,'-*','LineWidth', 2, 'color', 'k'); hold on;
   semilogy(x,cond_mgs_lvl2,'-+','LineWidth', 2, 'color', 'k'); hold on;
   grid on;
   axis([ min(x) max(x) 1e-17 1e100 ]);
   legend('Classical Gram-Schmidt', 'Classical Gram-Schmidt with Reorthogonalization', 'Delayed Classical Gram-Schmidt with Reorthogonalization', 'Householder - Level 1', 'Householder - Level 2', 'Modified Gram Schmidt - Level 1', 'Modified Gram Schmidt - Level 2','Location','northwest');
   title({'Condition Number of Krylov Basis, Arnoldi expansion','m = 500, k = 200'});
   xlabel('Arnoldi Expansion')
   ylabel('Condition Number, || [ b, A * Q ] ||_f_r_o')


