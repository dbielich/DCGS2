%
function [ mq, q, r ] = orth_mgs_lvl1_Ainnerprod( M, MQ, Q, a )
%  
      m = size(Q,1);
      j = size(Q,2)+1;
%
      r = zeros(j,1);
      q = zeros(m,1);
      mq = zeros(m,1);
%
      if j > 1
%
         for i = 1:j-1
%
            r(i,1) = a(1:m,1)' * MQ(1:m,i);   
            a(1:m,1) = a(1:m,1) - r(i,1) * Q(1:m,i);
%
         end
%
      end
%
         mq(1:m,1) = M(1:m,1:m) * a(1:m,1);
%
         tmp = a(1:m,1)' * mq(1:m,1);
         r(j,1) = sign( tmp ) * sqrt( abs( tmp ) );
         %r(j,1) = sqrt( abs( tmp ) );
%
         q(1:m,1) = a(1:m,1) / r(j,1);
         mq(1:m,1) = mq(1:m,1) / r(j,1);
%
   end
%
