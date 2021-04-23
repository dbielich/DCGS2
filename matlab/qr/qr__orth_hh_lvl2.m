% 
function [ Q, R ] = qr__orth_hh_lvl2( A )
%
   [m,n] = size(A);
%
   Q = zeros(m,n);
   V = zeros(m,n);
   R = zeros(n,n);
   T = zeros(n,n);
%      
   for j = 1:n,
%
      [ Q(1:m,j), T(1:j,j), R(1:j,j), V(1:m,j) ] = orth_hh_lvl2( V(1:m,1:j-1), T(1:j-1,1:j-1), A(1:m,j) );
%
   end
%
end
