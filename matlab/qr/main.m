%
   close all
   clear
%
   oldpath = path;
   path('../orth/',oldpath)  
   oldpath = path;
   path('../matrix_market/',oldpath)  
%
   m = 500;
   n = 100;
%
%  range_log10KA = 0:.5:16;
   range_log10KA = 0:1:16;
%
   condA           = zeros(size(range_log10KA));
%
   orth_cgs        = zeros(size(range_log10KA));
   orth_cgs2       = zeros(size(range_log10KA));
   orth_dcgs2      = zeros(size(range_log10KA));
   orth_hh_lvl1    = zeros(size(range_log10KA));
   orth_hh_lvl2    = zeros(size(range_log10KA));
   orth_mgs_lvl1   = zeros(size(range_log10KA));
   orth_mgs_lvl2   = zeros(size(range_log10KA));
%
   repres_cgs      = zeros(size(range_log10KA));
   repres_cgs2     = zeros(size(range_log10KA));
   repres_dcgs2    = zeros(size(range_log10KA));
   repres_hh_lvl1  = zeros(size(range_log10KA));
   repres_hh_lvl2  = zeros(size(range_log10KA));
   repres_mgs_lvl1 = zeros(size(range_log10KA));
   repres_mgs_lvl2 = zeros(size(range_log10KA));
%
   i = 0;
   for log10KA = range_log10KA;
      i = i+1;
%	   
      U = randn(m,n); [U,~]=qr(U,0);
      V = randn(n,n); [V,~]=qr(V,0);
      S = diag( 10.^( linspace( 0, log10KA, n ) ) );
      A = U * S * V';
      clear U S V;
      nrmA = norm(A,'fro');
      condA(i) = cond(A);
%
     [ Q, R ] = qr__orth_cgs              ( A ); [ repres_cgs(i),               orth_cgs(i),               s_cgs(i) ]               = qr__orth_stabilitymetric( A, Q, R ); clear Q R;
     [ Q, R ] = qr__orth_cgs2             ( A ); [ repres_cgs2(i),              orth_cgs2(i),              s_cgs2(i) ]              = qr__orth_stabilitymetric( A, Q, R ); clear Q R;
     [ Q, R ] = qr__orth_dcgs2            ( A ); [ repres_dcgs2(i),             orth_dcgs2(i),             s_dcgs2(i) ]             = qr__orth_stabilitymetric( A, Q, R ); clear Q R;
     [ Q, R ] = qr__orth_hh_lvl1          ( A ); [ repres_hh_lvl1(i),           orth_hh_lvl1(i),           s_hh_lvl1(i) ]           = qr__orth_stabilitymetric( A, Q, R ); clear Q R;
     [ Q, R ] = qr__orth_hh_lvl2          ( A ); [ repres_hh_lvl2(i),           orth_hh_lvl2(i),           s_hh_lvl2(i) ]           = qr__orth_stabilitymetric( A, Q, R ); clear Q R;
     [ Q, R ] = qr__orth_mgs_lvl1         ( A ); [ repres_mgs_lvl1(i),          orth_mgs_lvl1(i),          s_mgs_lvl1(i) ]          = qr__orth_stabilitymetric( A, Q, R ); clear Q R;
     [ Q, R ] = qr__orth_mgs_lvl2         ( A ); [ repres_mgs_lvl2(i),          orth_mgs_lvl2(i),          s_mgs_lvl2(i) ]          = qr__orth_stabilitymetric( A, Q, R ); clear Q R;
     [ Q, R ] = qr__orth_mgs_lvl1_backward( A ); [ repres_mgs_lvl1_backward(i), orth_mgs_lvl1_backward(i), s_mgs_lvl1_backward(i) ] = qr__orth_stabilitymetric( A, Q, R ); clear Q R;
     [ Q, R ] = qr__orth_mgs_lvl2_backward( A ); [ repres_mgs_lvl2_backward(i), orth_mgs_lvl2_backward(i), s_mgs_lvl2_backward(i) ] = qr__orth_stabilitymetric( A, Q, R ); clear Q R;
%
   end
%
   figure
   ax = axes;
   x = 10.^(range_log10KA);
   loglog(x,condA.^2*eps,'--','LineWidth', 2, 'color', 'k','HandleVisibility', 'off'); hold on;
   loglog(x,condA*eps,'--','LineWidth', 2, 'color', 'k','HandleVisibility', 'off'); hold on;
   loglog(x,orth_cgs,'-<','MarkerSize', 12, 'Color', '#eb8888', 'LineWidth', 2 ); hold on;
   loglog(x,orth_cgs2,'-<','MarkerSize', 12, 'Color', '#474747', 'LineWidth', 2 ); hold on;
   loglog(x,orth_dcgs2,'-s','MarkerSize', 12, 'Color', '#474747', 'LineWidth', 2 ); hold on;
   loglog(x,orth_hh_lvl1,'-^','MarkerSize', 12, 'Color', '#84aff5', 'LineWidth', 2 ); hold on;
   loglog(x,orth_hh_lvl2,'-s','MarkerSize', 12, 'Color', '#84aff5', 'LineWidth', 2 ); hold on;
   loglog(x,orth_mgs_lvl1,'-^','MarkerSize', 12, 'Color', '#f5e284', 'LineWidth', 2 ); hold on;
   loglog(x,orth_mgs_lvl2,'-s','MarkerSize', 12, 'Color', '#f5e284', 'LineWidth', 2 ); hold on;
%   loglog(x,orth_mgs_lvl1_backward,'-^','LineWidth', 2, 'color', '#de6f07'); hold on;
%   loglog(x,orth_mgs_lvl2_backward,'-+','LineWidth', 2, 'color', '#de6f07'); hold on;
   grid on;
   axis([ min(x) max(x) 1e-16 1e2 ]);
   legend('CGS', 'CGS2', 'DCGS2', 'HH lvl1', 'HH lvl2', 'MGS lvl1', 'MGS lvl2','Location','east');
   title({'Orthogonality Comparison, QR factorization','m = 500, n = 100'}, 'FontWeight', 'bold');
   xlabel('Condition Number of Input Matrix', 'FontWeight', 'bold')
   ylabel('Level of Orthogonality, || I - Q^T Q ||_f_r_o', 'FontWeight', 'bold')
   set(gca,'FontSize',12);   
   set(gca,'FontWeight','bold');   
%
   figure
   ax = axes;
   x = 10.^(range_log10KA);
   loglog(x,condA.^2*eps,'--','LineWidth', 2, 'color', 'k','HandleVisibility', 'off'); hold on;
   loglog(x,condA*eps,'--','LineWidth', 2, 'color', 'k','HandleVisibility', 'off'); hold on;
   loglog(x,repres_cgs,'-<','MarkerSize', 12, 'Color', '#eb8888', 'LineWidth', 2 ); hold on;
   loglog(x,repres_cgs2,'-<','MarkerSize', 12, 'Color', '#474747', 'LineWidth', 2 ); hold on;
   loglog(x,repres_dcgs2,'-s','MarkerSize', 12, 'Color', '#474747', 'LineWidth', 2 ); hold on;
   loglog(x,repres_hh_lvl1,'-^','MarkerSize', 12, 'Color', '#84aff5', 'LineWidth', 2 ); hold on;
   loglog(x,repres_hh_lvl2,'-s','MarkerSize', 12, 'Color', '#84aff5', 'LineWidth', 2 ); hold on;
   loglog(x,repres_mgs_lvl1,'-^','MarkerSize', 12, 'Color', '#f5e284', 'LineWidth', 2 ); hold on;
   loglog(x,repres_mgs_lvl2,'-s','MarkerSize', 12, 'Color', '#f5e284', 'LineWidth', 2 ); hold on;
   grid on;
   axis([ min(x) max(x) 1e-18 1e-2 ]);
   legend('CGS', 'CGS2', 'DCGS2', 'HH lvl1', 'HH lvl2', 'MGS lvl1', 'MGS lvl2','Location','east');
   title({'Representativity Comparison, QR factorization','m = 500, n = 100'}, 'FontWeight', 'bold');
   xlabel('Condition Number of Input Matrix', 'FontWeight', 'bold')
   ylabel('Level of Representativity, || A - Q R ||_f_r_o / || A ||_f_r_o', 'FontWeight', 'bold')
   set(gca,'FontSize',12);   
   set(gca,'FontWeight','bold');   

