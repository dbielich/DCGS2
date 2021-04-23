% 
function [ Q, H, beta ] = arnoldi__orth_hh_lvl1( A, b, k )
%
   [m] = size(A,1);
%
   Q = zeros(m,k);
   V = zeros(m,k);
   H = zeros(k,k-1);
   tau = zeros(k,1);
%      
   [ Q(1:m,1), tau(1), beta, V(1:m,1) ] = orth_hh_lvl1( V(1:m,1:0), tau(1:0,1), b(1:m,1) );
%
   for j = 2:k,
%
      V(1:m,j) = A * Q(1:m,j-1);
%
      [ Q(1:m,j), tau(j), H(1:j,j-1), V(1:m,j) ] = orth_hh_lvl1( V(1:m,1:j-1), tau(1:j-1,1), V(1:m,j) );
%
   end
%
end
