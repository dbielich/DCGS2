% 
function [ MQ, Q, R ] = qr__orth_mgs_lvl2_Ainnerprod_orth_projector_v2( A, M )
%
   [m,n] = size(A);
%
   Q = zeros(m,n);
   R = zeros(n,n);
   MQ = zeros(m,n);
   T = zeros(n,n);
%      
   for j = 1:n
%
      [ MQ(1:m,j), Q(1:m,j), T(1:j,j), R(1:j,j) ] = orth_mgs_lvl2_Ainnerprod_orth_projector_v2( M(1:m,1:m), MQ(1:m,1:j-1), Q(1:m,1:j-1), T(1:j-1,1:j-1), A(1:m,j) );
%
   end
%
end
