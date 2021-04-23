%
   function [ q, h ] = orth_dcgs2_arnoldi_cleanup( Q, H )
%  
      m = size(Q,1);
      k = size(Q,2); 
%
      work = zeros(k,1);
      h = zeros(k,1);
      q = zeros(m,1);
%
      work(1:k-1,1) = Q(1:m,1:k-1)' * Q(1:m,k);
      work(k,1)  = Q(1:m,k)' * Q(1:m,k);         
      h(k,1) = sqrt( work(k,1) - work(1:k-1,1)' * work(1:k-1,1) ); 
      q(1:m,1) = Q(1:m,k) - Q(1:m,1:k-1) * work(1:k-1,1);
      q(1:m,1) = q(1:m,1) / h(k,1);
      h(1:k-1,1) = H(1:k-1,k-1) + work(1:k-1,1);
%
   end 




