%
   function [ q, r ] = orth_cgs( Q, a )
%  
      m = size(Q,1);
      j = size(Q,2)+1;
%
      r = zeros(j,1);
      q = zeros(m,1);
%
      if j == 1,
         q(1:m,1) = a(1:m,1);
      end
%
      if j > 1,
         r(1:j-1,1) = Q(1:m,1:j-1)' * a(1:m,1);    
%    norm(r(1:j-1,1),2)	 
         q(1:m,1) = a(1:m,1) - Q(1:m,1:j-1) * r(1:j-1,1);
      end
%
      r(j,1) = norm( q(1:m,1) );                
      q(1:m,1) = q(1:m,1) / r(j,1);
%
   end
%
