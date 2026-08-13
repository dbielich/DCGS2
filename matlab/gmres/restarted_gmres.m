function [x, errors] = restarted_gmres(A, b, x0, max_iter, restart, tol)
    %
    oldpath = path;
    path('../orth/',oldpath)
    %
    % GMRES_CUSTOM Solves A*x = b using the Restarted GMRES method.
    % Inputs:
    %   A        - N x N matrix (or function handle)
    %   b        - N x 1 right-hand side vector
    %   x0       - N x 1 initial guess
    %   max_iter - Maximum number of total iterations
    %   restart  - Number of iterations before restart (m)
    %   tol      - Relative residual tolerance
    % Outputs:
    %   x        - Approximate solution vector
    %   errors   - Vector of residual norms at each iteration

    n = length(b);

    x = x0;
    r = b - A*x;
    beta = norm(r);
    errors = [beta];

    if beta < tol * norm(b)
        return;
    end

    iter = 0;
    while iter < max_iter
        % Allocate Krylov basis V and Hessenberg matrix H
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

        % Arnoldi Process
        for j = 1:m
            iter = iter + 1 ;
            w = A * Q(:, j) ;

            %[ Q(:, j + 1), H(:, j) ] = orth_mgs_lvl1( Q(:, 1:j), w ) ;
            %[ Q(:, j + 1), T(:, j + 1), H(:, j) ] = orth_mgs_lvl2( Q(:, 1:j), T(1:j, 1:j), w ) ;
            [ Q(:, j + 1), tau(j + 1, 1), H(:, j), V(:, j + 1) ] = orth_hh_lvl1( V(:, 1:j), tau(1:j, 1), w ) ;

            % Solve the least squares problem min ||H*y - g|| using Givens rotations
            % (Using MATLAB's backslash for simplicity in this section)
            H_sub = H(1:j+1, 1:j) ;
            g_sub = g(1:j+1) ;
            y = H_sub \ g_sub ;


            % Check current residual without explicitly forming x
            res_norm = norm(H_sub * y - g_sub) ;
            errors = [errors; res_norm] ;

            if res_norm < tol * norm(b)
                m = j ;
                break ;
            end
        end

        % Update solution
        y = H(1:m+1, 1:m) \ g(1:m+1) ;
        x = x + Q(:, 1:m) * y ;

        % Check global convergence
        if errors(end) < tol * norm(b)
            break;
        end
    end
end

