%
   function [ q, r ] = orth_mgs_lvl1_backward( Q, a )
%  
      m = size(Q,1);
      j = size(Q,2)+1;
%
      r = zeros(j,1);
%
      for i = j-1:-1:1,
         r(i,1) = Q(1:m,i)' * a(1:m,1);    %%%%%%%%% <- j-1 synchronizations
         a(1:m,1) = a(1:m,1) - Q(1:m,i) * r(i,1);
      end
%
         r(j,1) = norm( a(1:m,1) );        %%%%%%%%% <- 1 synchronization
         q(1:m,1) = a(1:m,1) / r(j,1);
%
   end
%
