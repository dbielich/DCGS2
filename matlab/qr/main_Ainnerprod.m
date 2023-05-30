%
   close all
   clear
%
   oldpath = path;
   path('../orth/',oldpath)  
   oldpath = path;
   path('../matrix_market/',oldpath)  
%
   m = 100;
   n = 50;
%
   M = rand(m);
   M = 0.5 * (M + M');
   M = M + (m * eye(m));
%
   range_log10KA = 0:1:16;
%
   condA               = zeros(size(range_log10KA));
%
   orth_mgs_lvl1       = zeros(size(range_log10KA));
   orth_mgs_lvl2       = zeros(size(range_log10KA));
   orth_mgs_lvl2_2sync = zeros(size(range_log10KA));
   orth_mgs_lvl2_opv1 = zeros(size(range_log10KA));
   orth_mgs_lvl2_opv2 = zeros(size(range_log10KA));
%
   repres_mgs_lvl1       = zeros(size(range_log10KA));
   repres_mgs_lvl2       = zeros(size(range_log10KA));
   repres_mgs_lvl2_2sync = zeros(size(range_log10KA));
   repres_mgs_lvl2_opv1 = zeros(size(range_log10KA));
   repres_mgs_lvl2_opv2 = zeros(size(range_log10KA));
%
   s_mgs_lvl1            = zeros(size(range_log10KA));
   s_mgs_lvl2            = zeros(size(range_log10KA));
   s_mgs_lvl2_2sync      = zeros(size(range_log10KA));
   s_mgs_lvl2_opv1      = zeros(size(range_log10KA));
   s_mgs_lvl2_opv2      = zeros(size(range_log10KA));
%
   i = 0;
   for log10KA = range_log10KA
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
     [ MQ, Q, R ] = qr__orth_mgs_lvl1_Ainnerprod( A, M ); [ repres_mgs_lvl1(i), orth_mgs_lvl1(i), s_mgs_lvl1(i) ] = qr_Ainnerprod_orth_stabilitymetric( A, M, Q, R ); clear MQ Q R;
     [ MQ, Q, R ] = qr__orth_mgs_lvl2_Ainnerprod( A, M ); [ repres_mgs_lvl2(i), orth_mgs_lvl2(i), s_mgs_lvl2(i) ] = qr_Ainnerprod_orth_stabilitymetric( A, M, Q, R ); clear MQ Q R;
     [ MQ, Q, R ] = qr__orth_mgs_lvl2_Ainnerprod_2sync( A, M ); [ repres_mgs_lvl2_2sync(i), orth_mgs_lvl2_2sync(i), s_mgs_lvl2_2sync(i) ] = qr_Ainnerprod_orth_stabilitymetric( A, M, Q, R ); clear MQ Q R;
     [ MQ, Q, R ] = qr__orth_mgs_lvl2_Ainnerprod_orth_projector( A, M ); [ repres_mgs_lvl2_opv1(i), orth_mgs_lvl2_opv1(i), s_mgs_lvl2_opv1(i) ] = qr_Ainnerprod_orth_stabilitymetric( A, M, Q, R ); clear MQ Q R;
     [ MQ, Q, R ] = qr__orth_mgs_lvl2_Ainnerprod_orth_projector_v2( A, M ); [ repres_mgs_lvl2_opv2(i), orth_mgs_lvl2_opv2(i), s_mgs_lvl2_opv2(i) ] = qr_Ainnerprod_orth_stabilitymetric( A, M, Q, R ); clear MQ Q R;
%
   end
%
   figure
   ax = axes;
   x = 10.^(range_log10KA);
   loglog(x,condA.^2*eps,'--','LineWidth', 2, 'color', 'k','HandleVisibility', 'off'); hold on;
   loglog(x,condA*eps,'--','LineWidth', 2, 'color', 'k','HandleVisibility', 'off'); hold on;
   loglog(x,orth_mgs_lvl1,'-s','MarkerSize', 12, 'Color', '#f5e284', 'LineWidth', 2 ); hold on;
   loglog(x,orth_mgs_lvl2,'-s','MarkerSize', 12, 'Color', '#4D0F6D', 'LineWidth', 2 ); hold on;
   loglog(x,orth_mgs_lvl2_2sync,'-s','MarkerSize', 12, 'Color', '#0F6D19', 'LineWidth', 2 ); hold on;
   loglog(x,orth_mgs_lvl2_opv1,'-s','MarkerSize', 12, 'Color', '#D0761B', 'LineWidth', 2 ); hold on;
   loglog(x,orth_mgs_lvl2_opv2,'-s','MarkerSize', 12, 'Color', '#996633', 'LineWidth', 2 ); hold on;
   grid on;
   axis([ min(x) max(x) 1e-16 1e2 ]);
   legend('MGS lvl1','MGS lvl2','MGS lvl2 2-sync','MGS lvl2 Orthogonal Projection V1','MGS lvl2 Orthogonal Projection V2','Location','southeast');
   title({'Orthogonality Comparison, QR factorization','m = 500, n = 100'}, 'FontWeight', 'bold');
   xlabel('Condition Number of Input Matrix', 'FontWeight', 'bold')
   ylabel('Level of Orthogonality, || I - Q^T Q ||_f_r_o', 'FontWeight', 'bold')
   set(gca,'FontSize',12);   
   set(gca,'FontWeight','bold');   
%
return
   figure
   ax = axes;
   x = 10.^(range_log10KA);
   loglog(x,condA.^2*eps,'--','LineWidth', 2, 'color', 'k','HandleVisibility', 'off'); hold on;
   loglog(x,condA*eps,'--','LineWidth', 2, 'color', 'k','HandleVisibility', 'off'); hold on;
   loglog(x,repres_mgs_lvl1,'-s','MarkerSize', 12, 'Color', '#f5e284', 'LineWidth', 2 ); hold on;
   loglog(x,repres_mgs_lvl2,'-s','MarkerSize', 12, 'Color', '#4D0F6D', 'LineWidth', 2 ); hold on;
   loglog(x,repres_mgs_lvl2_2sync,'-s','MarkerSize', 12, 'Color', '#0F6D19', 'LineWidth', 2 ); hold on;
   grid on;
   axis([ min(x) max(x) 1e-17 1e2 ]);
   legend('MGS lvl1','MGS lvl2','MGS lvl2 2-sync','Location','east');
   title({'Representativity Comparison, QR factorization','m = 500, n = 100'}, 'FontWeight', 'bold');
   xlabel('Condition Number of Input Matrix', 'FontWeight', 'bold')
   ylabel('Level of Representativity, || A - Q R ||_f_r_o / || A ||_f_r_o', 'FontWeight', 'bold')
   set(gca,'FontSize',12);   
   set(gca,'FontWeight','bold');   
%
