%
   function [ q, t, r, v ] = orth_hh_lvl2( V, T, a )
%
      m = size(V,1);
      j = size(V,2)+1;
%
      q = zeros(m,1);
      v = zeros(m,1);
      r = zeros(j,1);
      t = zeros(j,1);
%
      if (j > 1 )
         r(1:j-1,1) = V(1:m,1:j-1)' * a(1:m,1);  
         r(1:j-1,1) = T(1:j-1,1:j-1)' * r(1:j-1,1);
         a(1:m,1) = a(1:m,1) - V(1:m,1:j-1) * r(1:j-1,1);
%
         r(1:j-1,1) = a(1:j-1,1);                   
      end
%
      normx = norm( a(j+1:m,1) , 2);            
      norma = sqrt( normx*normx +  a(j,1)*a(j,1) );
      if ( a(j,1) > 0.0e+00 ) v(j,1) = a(j,1) + norma; else v(j,1) = a(j,1) - norma; end
      v(j+1:m,1) = a(j+1:m,1) / v(j,1);
      if ( a(j,1) > 0.0e+00 ) r(j,1) = - norma; else r(j,1) = norma; end
      tau = 2.0e+00 / ( 1.0e+00 + ( normx / v(j,1) )^2 );
      v(j,1) = 1.0e+00;
      v(1:j-1,1) = 0.0e+00;
%
      q(j,1) = - tau;
      q(j+1:m,1) = v(j+1:m,1) * q(j,1) ;
      q(j,1) = 1.0e+00 + q(j,1) ;
      if (j > 1 )
         q(1:j-1,1) = 0.e+00;
         t(1:j-1,1) = V(1:m,1:j-1)' * q(1:m,1) ; 
         t(1:j-1,1) = T(1:j-1,1:j-1) * t(1:j-1,1) ;
         q(1:m,1) = q(1:m,1) - V(1:m,1:j-1) * t(1:j-1,1) ;
      end
%
      if (j > 1 )
         t(1:j-1,1) = V(1:m,1:j-1)' * v(1:m,1);       
         t(1:j-1,1) = -t(1:j-1,1) * tau;
         t(1:j-1,1) = T(1:j-1,1:j-1) * t(1:j-1,1);
      end
      t(j,1) = tau;
%
   end
%

