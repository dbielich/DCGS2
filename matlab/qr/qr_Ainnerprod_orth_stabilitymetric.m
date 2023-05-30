%
function [ repres, orth, normS ] = qr_Ainnerprod_orth_stabilitymetric( A, M, Q, R )
%
   [m,n] = size(A);
%
   if( (size(A,1) ~= m) & (size(A,2) ~= n) )   fprintf("ERROR1 in arnoli__orth_stabilitymetric\n"); end;
   if( (size(Q,1) ~= m) & (size(Q,2) ~= n) )   fprintf("ERROR2 in arnoli__orth_stabilitymetric\n"); end;
   if( (size(R,1) ~= n) & (size(R,2) ~= n) )   fprintf("ERROR3 in arnoli__orth_stabilitymetric\n"); end;
%
   nrmA       = norm(A,'fro');
%
   orth       = norm( eye(n) - Q(1:m,1:n)' * M(1:m,1:m) * Q(1:m,1:n), 'fro');
%
   repres     = norm( A(1:m,1:n) - Q(1:m,1:n) * R(1:n,1:n), 'fro') / nrmA;
%
   U(1:n,1:n) = triu(Q(1:m,1:n)' * M(1:m,1:m) * Q(1:m,1:n), 1);
   S          = (eye(n) + U) \ U;
   normS      = norm(S,2);
%
end
%
