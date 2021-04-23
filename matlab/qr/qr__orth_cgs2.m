% 
function [ Q, R ] = qr__orth_cgs2( A )
%
   [m,n] = size(A);
%
   Q = zeros(m,n);
   R = zeros(n,n);
%      
   for j = 1:n,
%
      [ Q(1:m,j), R(1:j,j) ] = orth_cgs2( Q(1:m,1:j-1), A(1:m,j) );
%
   end
%
end
