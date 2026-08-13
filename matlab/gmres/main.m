close all
clear all

oldpath = path;
path('../orth/',oldpath)

% 1. Create a large, non-symmetric sparse matrix
n = 100;
A = sprand(n, n, 0.1) + 2 * speye(n);
b = rand(n, 1);
x0 = zeros(n, 1);

% 2. Set Parameters
max_iter = 120;
restart = 1;
tol = 1e-6;

% 3. Run Custom GMRES
[x, errors] = restarted_gmres(A, b, x0, max_iter, restart, tol);

% 4. Verify against MATLAB's built-in GMRES
[x_matlab, flag, relres] = gmres(A, b, restart, tol, max_iter/restart);

% 5. Print results
fprintf('Difference between custom and built-in: %e\n', norm(x - x_matlab));

% 6. Plot Convergence
figure;
semilogy(0:length(errors)-1, errors/norm(b), '-o', 'LineWidth', 1.5);
grid on;
title('GMRES Convergence History');
xlabel('Iteration Count');
ylabel('Relative Residual Norm');
