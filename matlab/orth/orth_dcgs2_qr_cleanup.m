%
   function [ q, r ] = orth_dcgs2_qr_cleanup( Q, R )
%  
      m = size(Q,1);
      n = size(Q,2); 
%
      r = zeros(n,1);
      q = zeros(m,1);
%
      r(1:n,1) = Q(1:m,1:n)' * Q(1:m,n); 
% 
      r(n,1) = r(n,1) - r(1:n-1,1)' * r(1:n-1,1);
%
      r(n,1) = sqrt( r(n,1) );  
%      
      Q(1:m,n) = Q(1:m,n) - Q(1:m,1:n-1) * r(1:n-1,1);  
%
      r(1:n-1,1) = R(1:n-1,1) + r(1:n-1,1);  
%
      q(1:m,1) = Q(1:m,n) / r(n,1); 
%
   end 





