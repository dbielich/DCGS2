%
   function [ q, r ] = orth_mgs_lvl1( Q, a )
%  
      m = size(Q,1);
      j = size(Q,2)+1;
%
      r = zeros(j,1);
      q = zeros(m,1);
%
      if j == 1,
%
         q(1:m,1) = a(1:m,1);	 
%
      end
%
      if j > 1,
%
         for i = 1:j-1,
%
            r(i,1) = Q(1:m,i)' * a(1:m,1);   
            a(1:m,1) = a(1:m,1) - Q(1:m,i) * r(i,1);
%
         end
%
      end
%
         r(j,1) = norm( a(1:m,1) );       
         q(1:m,1) = a(1:m,1) / r(j,1);
%
   end
%
