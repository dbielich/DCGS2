% 
function [ MQ, Q, R ] = qr__orth_mgs_lvl1_Ainnerprod( A, M )
%
   [m,n] = size(A);
%
   Q = zeros(m,n);
   R = zeros(n,n);
   MQ = zeros(m,n);
%      
   for j = 1:n
%
      [ MQ(1:m,j), Q(1:m,j), R(1:j,j) ] = orth_mgs_lvl1_Ainnerprod( M(1:m,1:m), MQ(1:m,1:j-1), Q(1:m,1:j-1), A(1:m,j) );
%
   end
%
end
