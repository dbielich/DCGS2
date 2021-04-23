% 
function [ Q, R ] = qr__orth_dcgs2( A )
%
   [m,n] = size(A);
%
   Q = A;
   R = zeros(n,n);
%
   for j = 2:n,
%
      [ Q(1:m,j-1:j), R(1:j-1,j-1:j) ] = orth_dcgs2_qr( Q(1:m,1:j), R(1:j-2,j-1) );
%
   end
%
   [ Q(1:m,n), R(1:n,n) ] = orth_dcgs2_qr_cleanup( Q(1:m,1:n), R(1:n-1,n) );      
%   
end
