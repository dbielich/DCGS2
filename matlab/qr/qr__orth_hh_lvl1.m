% 
function [ Q, R ] = qr__orth_hh_lvl1( A )
%
   [m,n] = size(A);
%
   Q = zeros(m,n);
   V = zeros(m,n);
   R = zeros(n,n);
   tau = zeros(n,1);
%      
   for j = 1:n,
%
      [ Q(1:m,j), tau(j), R(1:j,j), V(1:m,j) ] = orth_hh_lvl1( V(1:m,1:j-1), tau(1:j-1,1), A(1:m,j) );
%
   end
%
end
