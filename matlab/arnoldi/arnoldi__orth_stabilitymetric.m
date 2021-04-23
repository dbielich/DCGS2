%
function [ repres, orth, condn, normS ] = arnoldi__orth_stabilitymetric( A, b, Q, H, beta )
%
   [m] = size(A,1);
   [k] = size(H,1);
%
   if( (size(A,1) ~= m) & (size(A,2) ~= m) )   fprintf("ERROR1 in arnoli__orth_stabilitymetric\n"); end;
   if( (size(b,1) ~= m) & (size(b,2) ~= 1) )   fprintf("ERROR2 in arnoli__orth_stabilitymetric\n"); end;
   if( (size(Q,1) ~= m) & (size(Q,2) ~= k) )   fprintf("ERROR3 in arnoli__orth_stabilitymetric\n"); end;
   if( (size(H,1) ~= k) & (size(H,2) ~= k-1) ) fprintf("ERROR4 in arnoli__orth_stabilitymetric\n"); end;
%
   repres = zeros(1,k);
   orth   = zeros(1,k);
   condn  = zeros(1,k);
   normS  = zeros(1,k);
%
   nrmA = norm(A,'fro');
%
   for j = 1:k,
%
      orth(j)   = norm( eye(j) - Q(1:m,1:j)' * Q(1:m,1:j), 'fro');
%
      repres(j) = norm( [ b, A * Q(1:m,1:j-1) ] - Q(1:m,1:j) * [ [beta; zeros(j-1,1) ], H(1:j,1:j-1)], 'fro') / nrmA;
%
      condn(j)  = cond( [ b, A * Q(1:m,1:j-1) ] );
%
      U(1:j,1:j) = triu(Q(1:m,1:j)' * Q(1:m,1:j), 1);
      S = (eye(j) + U) \ U;
      normS(j) = norm(S,2);
%
   end
%
end
%
