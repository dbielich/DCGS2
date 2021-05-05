%
   function [ q, t, r ] = orth_mgs_lvl2( Q, T, a )
%  
      m = size(Q,1);
      j = size(Q,2)+1;
%
      r = zeros(j,1);
      t = zeros(j,1);
      q = zeros(j,1);
%
      if j == 1 ,
%
	 q(1:m,1) = a(1:m,1);
	 t(1,1) = 1.0e+00;
%
      end
%
      if j > 1 ,
%
         r(1:j-1,1) = Q(1:m,1:j-1)' * a(1:m,1);   
	 r(1:j-1,1) = T(1:j-1,1:j-1)' * r(1:j-1,1);
         q(1:m,1) = a(1:m,1) - Q(1:m,1:j-1) * r(1:j-1,1);
%
      end  
%
      r(j,1) = norm( q(1:m,1) );
      q(1:m,1) = q(1:m,1) / r(j,1);
%
      if j > 1 ,
%
         t(1:j-1,1) = - Q(1:m,1:j-1)' * q(1:m,1);
         t(1:j-1,1) = T(1:j-1,1:j-1) * t(1:j-1,1);
         t(j,1) = 1.0e+00;
%
      end    
%
   end
%
