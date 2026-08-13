include("restarted_gmres.jl")
include("relaxed_gmres.jl")
using LinearAlgebra, SparseArrays, Plots

demo_nilpotent = false  # bidiagonal nilpotent: too easy for GMRES(m), restarts recover cheaply
demo_grcar     = true   # Grcar matrix: reliable stagnation due to pseudospectrum surrounding origin

if demo_grcar
    n  = 60
    # non-normal Toeplitz: eigenvalues in an annulus, pseudospectrum wraps around 0
    A  = diagm(0  =>  ones(n),
               1  =>  ones(n-1),
               2  =>  ones(n-2),
               3  =>  ones(n-3),
               -1 => -ones(n-1))
    b  = randn(n)
    x0 = zeros(n)
    tol = 1e-10

    _, err_full  = restarted_gmres(A, b, x0, n*3,  n, tol)
    _, err_r5    = restarted_gmres(A, b, x0, n*30, 5, tol)
    _, err_r15   = restarted_gmres(A, b, x0, n*10, 15, tol)

    println("Full GMRES  final residual: ", err_full[end]  / norm(b), " (", length(err_full)-1,  " iters)")
    println("GMRES(5)    final residual: ", err_r5[end]    / norm(b), " (", length(err_r5)-1,    " iters)")
    println("GMRES(15)   final residual: ", err_r15[end]   / norm(b), " (", length(err_r15)-1,   " iters)")

    p = plot(err_full ./ norm(b),  yscale=:log10, label="Full GMRES (m=n)", lw=2)
    plot!(p, err_r15 ./ norm(b),   label="GMRES(15)",                       lw=2)
    plot!(p, err_r5  ./ norm(b),   label="GMRES(5)",                        lw=2)
    xlabel!(p, "Iteration Count")
    ylabel!(p, "Relative Residual Norm")
    title!(p, "Grcar matrix: GMRES(m) vs full GMRES")
    display(p)

elseif demo_nilpotent
    n       = 80
    A       = I(n) + 0.5 * diagm(1 => ones(n-1))   # nilpotent shift: λ=1, order n
    b       = randn(n)
    x0      = zeros(n)
    tol     = 1e-10

    _, err_full      = restarted_gmres(A, b, x0, n,    n,  tol)
    _, err_restarted = restarted_gmres(A, b, x0, n*5, 10,  tol)

    println("Full GMRES   final residual: ", err_full[end]      / norm(b), " (", length(err_full)-1,      " iters)")
    println("GMRES(10)    final residual: ", err_restarted[end]  / norm(b), " (", length(err_restarted)-1, " iters)")

    p = plot(err_full ./ norm(b),       yscale=:log10, label="Full GMRES (m=n)", lw=2, marker=:circle)
    plot!(p, err_restarted ./ norm(b),  label="GMRES(10)",                       lw=2, marker=:square)
    xlabel!(p, "Iteration Count")
    ylabel!(p, "Relative Residual Norm")
    title!(p, "Nilpotent shift: GMRES(m) vs full GMRES")
    display(p)
else
    n = 100
    A = sprand(n, n, 0.1) + 2I
    b = rand(n)
    x0 = zeros(n)

    max_iter = 120
    restart  = 30
    tol      = 1e-6

    x, errors = restarted_gmres(A, b, x0, max_iter, restart, tol)
    println("Restarted GMRES residual: ", norm(b .- A * x) / norm(b))

    x_relax, errors_relax, formats = relaxed_gmres(A, b, x0, max_iter, restart, tol, 0.5)
    println("Relaxed  GMRES residual: ", norm(b .- A * x_relax) / norm(b))
    println("Formats used: ", unique(formats))

    p = plot(0:length(errors)-1, errors ./ norm(b),
             yscale=:log10, label="Restarted (Float64)", lw=2, marker=:circle)
    plot!(p, 0:length(errors_relax)-1, errors_relax ./ norm(b),
          label="Relaxed (adaptive)", lw=2, marker=:square)
    xlabel!(p, "Iteration Count")
    ylabel!(p, "Relative Residual Norm")
    title!(p, "GMRES Convergence History")
    display(p)
end

