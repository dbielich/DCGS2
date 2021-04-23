%
   function [ q, r, h ] = orth_dcgs2_arnoldi( Q, H )
%  
      m = size(Q,1);
      j = size(Q,2); 
%
      h = zeros(j-1,1);
      r = zeros(j-1,1);
      q = zeros(m,2);
%
      if( j == 2 ),
%
         r(j-1,j-1)  = Q(1:m,j-1)' * Q(1:m,j);
	 q(1:m,j-1) = Q(1:m,j-1);
         q(1:m,j) = Q(1:m,j) - Q(1:m,1:j-1) * r(j-1,j-1);
%
      end
%
      if( j > 2 ),
%
         work(1:j-1,1:2) = Q(1:m,1:j-1)' *  Q(1:m,j-1:j);
%
%    norm(work(1:j-1,1),2)
%
	 h(j-1,1) = sqrt( work(j-1,1) - work(1:j-2,1)' * work(1:j-2,1) );
         work(j-1,2) = ( work(j-1,2) - work(1:j-2,1)' * work(1:j-2,2) ) / ( h(j-1,1) * h(j-1,1) );
         work(1:j-2,2) = work(1:j-2,2) / h(j-1,1);
%
         q(1:m,1) = Q(1:m,j-1) - Q(1:m,1:j-2) * work(1:j-2,1);
         q(1:m,1) = q(1:m,1) / h(j-1,1);
         q(1:m,2) = ( Q(1:m,j) / h(j-1,1) ) - [ Q(1:m,1:j-2), q(1:m,1) ] * work(1:j-1,2);
%
         h(1:j-2,1) = H(1:j-2,j-2) + work(1:j-2,1);      
         r(1:j-1,1) = work(1:j-1,2) - ( H(1:j-1,1:j-2) * work(1:j-2,1) ) / h(j-1,1);
%
      end
%
   end 





