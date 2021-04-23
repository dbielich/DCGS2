% 
function [ Q, H, beta ] = arnoldi__orth_cgs2( A, b, k )
%
   [m] = size(A,1);
%
   Q = zeros(m,k);
   H = zeros(k,k-1);
%
   [ Q(1:m,1), beta ] = orth_cgs2( Q(1:m,1:0), b(1:m,1) );
%
   for j = 2:k,
%
      Q(1:m,j) = A * Q(1:m,j-1);
%
      [ Q(1:m,j), H(1:j,j-1) ] = orth_cgs2( Q(1:m,1:j-1), Q(1:m,j) );
%
   end
%
end
