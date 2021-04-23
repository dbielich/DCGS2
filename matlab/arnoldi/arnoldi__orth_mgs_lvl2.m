% 
function [ Q, H, beta ] = arnoldi__orth_mgs_lvl2( A, b, k )
%
   [m] = size(A,1);
%
   Q = zeros(m,k);
   H = zeros(k,k-1);
   T = zeros(k,k);
%      
   beta = norm(b);
   Q(1:m,1) = b / beta;
   T(1,1) = 1.0e+00;
%
   for j = 2:k,
%
      Q(1:m,j) = A * Q(1:m,j-1);
%
      [ Q(1:m,j), T(1:j,j), H(1:j,j-1) ] = orth_mgs_lvl2( Q(1:m,1:j-1), T(1:j-1,1:j-1), Q(1:m,j) );
%
   end
%
end
