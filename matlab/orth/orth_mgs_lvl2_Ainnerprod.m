%
function [ mq, q, t, r ] = orth_mgs_lvl2_Ainnerprod( M, MQ, Q, T, a )
%  
      m = size(Q,1);
      j = size(Q,2)+1;
%
      r = zeros(j,1);
      q = zeros(m,1);
      t = zeros(j,1);
      mq = zeros(m,1);
%
      if j > 1
%
         r(1:j-1,1) = a(1:m,1)' * MQ(1:m,1:j-1);   
         r(1:j-1,1) = T(1:j-1,1:j-1)' * r(1:j-1,1);
         a(1:m,1) = a(1:m,1) - Q(1:m,1:j-1) * r(1:j-1,1);
%
      end
%
      mq(1:m,1) = M(1:m,1:m) * a(1:m,1);
%
      tmp = a(1:m,1)' * mq(1:m,1);
      r(j,1) = sign( tmp ) * sqrt( abs( tmp ) );
      t(j,1) = 1.0e+00;
%
      q(1:m,1) = a(1:m,1) / r(j,1);
      mq(1:m,1) = mq(1:m,1) / r(j,1);
%
      if j > 1
%
         t(1:j-1,1) = - MQ(1:m,1:j-1)' * q(1:m,1);
         t(1:j-1,1) = T(1:j-1,1:j-1) * t(1:j-1,1);
%
      end   
%
   end
%
