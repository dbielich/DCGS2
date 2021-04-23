%
   function [ q, t, r, a ] = orth_hh_lvl1( V, tau, a ) 
%  
      m = size(V,1);
      j = size(V,2)+1;
%
      r = zeros(j,1);
      q = zeros(m,1);
%
      for i = 1:j-1,
         alpha = tau(i) * ( V(i:m,i)' * a(i:m,1) );       
         a(i:m,1) = a(i:m,1) - V(i:m,i) * alpha;          
      end 
%
      if (j>1), r(1:j-1,1) = a(1:j-1,1); end;           
%
      normx = norm( a(j+1:m,1), 2);                       
      norma = sqrt( normx*normx +  a(j,1)*a(j,1) );
      if ( a(j,1) > 0.0e+00 ) r(j,1) = - norma; else r(j,1) = norma; end
      if ( a(j,1) > 0.0e+00 ) a(j,1) = a(j,1) + norma; else a(j,1) = a(j,1) - norma; end
      a(j+1:m,1) = a(j+1:m,1) / a(j,1);
      t = 2.0e+00 / ( 1.0e+00 + ( normx / a(j,1) )^2 );
      a(1:j-1,1) = 0.0e+00;
      a(j,1) = 1.0e+00;
%
      q(j,1) = - t;
      q(j+1:m,1) = a(j+1:m,1) * q(j,1);
      q(j,1) = 1.0e+00 + q(j,1);
      for i = j-1:-1:1,
        q(i,1) = - tau(i) * ( V(i+1:m,i)' * q(i+1:m,1) );  
        q(i+1:m,1) = q(i+1:m,1) + V(i+1:m,i) * q(i,1);     
      end
%
   end

