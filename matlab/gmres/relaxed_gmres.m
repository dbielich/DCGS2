function [x, errors, formats] = relaxed_gmres(A, b, x0, max_iter, restart, tol, eta)
    % Inexact restarted GMRES with adaptive-precision matvec.
    %
    % Inputs:
    %   A        - N x N matrix (or function handle)
    %   b        - N x 1 right-hand side vector
    %   x0       - N x 1 initial guess
    %   max_iter - Maximum number of total iterations
    %   restart  - Number of iterations before restart (m)
    %   tol      - Relative residual tolerance
    %   eta      - Relaxation strength in [0,1]: 0 = pure double, 1 = max relaxation.
    %              Requires chop (github.com/higham/chop) when eta > 0.
    % Outputs:
    %   x        - Approximate solution vector
    %   errors   - Vector of residual norms at each iteration
    %   formats  - Format used for the matvec at each iteration ('h','s','d')
    %
    % Theory: Simoncini & Szyld (2003), SIAM J. Sci. Comput. 25(2), 454-477.
    %         Error bound: ||f_j|| <= eta_j * ||A*q_j||, eta_j = eta * rel_res_j.

    oldpath = path;
    path('../orth/', oldpath)

    n = length(b);
    nrm_b = norm(b);
    x = x0;
    r = b - A*x;
    beta = norm(r);
    errors = [beta];
    formats = {};

    if beta < tol * nrm_b
        return;
    end

    iter = 0;
    res_prev = beta;

    while iter < max_iter
        m = min(restart, max_iter - iter) ;
        Q = zeros(n, m + 1) ;
        V = zeros(n, m + 1) ;
        H = zeros(m + 1, m) ;
        tau = zeros(m + 1, 1) ;

        r = b - A*x ;
        beta = norm(r) ;
        [ Q(:,1), tau(1), h_init, V(:,1) ] = orth_hh_lvl1( V(:,1:0), tau(1:0,1), r ) ;

        g = zeros(m + 1, 1) ;
        g(1) = h_init(1) ;

        for j = 1:m
            iter = iter + 1 ;

            [ w, fmt ] = matvec_relaxed(A, Q(:,j), eta, res_prev / nrm_b) ;
            formats{end+1} = fmt ;

            [ Q(:, j+1), tau(j+1), H(:,j), V(:,j+1) ] = orth_hh_lvl1( V(:,1:j), tau(1:j,1), w ) ;

            H_sub = H(1:j+1, 1:j) ;
            g_sub = g(1:j+1) ;
            y = H_sub \ g_sub ;

            res_norm = norm(H_sub * y - g_sub) ;
            errors = [errors; res_norm] ;
            res_prev = res_norm ;

            if res_norm < tol * nrm_b
                m = j ;
                break ;
            end
        end

        y = H(1:m+1, 1:m) \ g(1:m+1) ;
        x = x + Q(:, 1:m) * y ;

        if errors(end) < tol * nrm_b
            break;
        end
    end
end

function [w, fmt] = matvec_relaxed(A, q, eta, rel_res)
% Compute A*q at the coarsest precision within the error budget eta*rel_res.
%
% Format unit roundoffs:
%   fp16   (h): u ~ 2^{-11} ~ 4.88e-4
%   single (s): u ~ 2^{-24} ~ 5.96e-8
%   double (d): u ~ 2^{-53} ~ 1.11e-16
    budget = eta * rel_res ;
    if budget >= 4.88e-4
        fmt = 'h' ;
    elseif budget >= 5.96e-8
        fmt = 's' ;
    else
        w = A * q ;
        fmt = 'd' ;
        return ;
    end
    w = chop(A * q, struct('format', fmt)) ;
end
