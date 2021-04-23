% 
function [ Q, H, beta ] = arnoldi__orth_dcgs2( A, b, k )
%
   [m] = size(A,1);
%
   Q = zeros(m,k);
   H = zeros(k,k-1);
%
   nrmA = norm(A,'fro');
   beta = norm(b);
   Q(1:m,1) = b / beta;
%
   for j = 2:k,
%
      Q(1:m,j) = A * Q(1:m,j-1);
%
      if( j == 2 ), [ Q(1:m,j-1:j), H(1:j-1,j-1), do_not_need ]  = orth_dcgs2_arnoldi( Q(1:m,1:j), H(1:j-1,1:j-2) ); end
      if( j >= 3 ), [ Q(1:m,j-1:j), H(1:j-1,j-1), H(1:j-1,j-2) ] = orth_dcgs2_arnoldi( Q(1:m,1:j), H(1:j-1,1:j-2) ); end
%
   end
%
   [ Q(1:m,k), H(1:k,k-1) ] = orth_dcgs2_arnoldi_cleanup( Q(1:m,1:k), H(1:k-1,1:k-1) );      
%   
end
