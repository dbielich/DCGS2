include("restarted_gmres.jl")
include("relaxed_gmres.jl")
include("relaxed_gmres_low.jl")
using LinearAlgebra, SparseArrays, Plots, Random

# count occurrences of each format without StatsBase
countmap(v) = Dict(k => count(==(k), v) for k in unique(v))

demo_nilpotent = false  # bidiagonal nilpotent: too easy for GMRES(m), restarts recover cheaply
demo_grcar     = false  # Grcar matrix: reliable stagnation due to pseudospectrum surrounding origin

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
    Random.seed!(42)
    n  = 60
    # Symmetric positive definite: eigenvalues log-spaced in [1, 100], cond = 100.
    # Full GMRES converges in ~90 steps; GMRES(15) in ~150-250 steps.
    F  = qr(randn(n, n))
    Q  = Matrix(F.Q)
    S  = Diagonal(10 .^ LinRange(0, 2, n))
    A  = Q * Matrix(S) * Q'
    
    n  = 60
    # non-normal Toeplitz: eigenvalues in an annulus, pseudospectrum wraps around 0
    A  = diagm(0  =>  ones(n),
               1  =>  ones(n-1),
               2  =>  ones(n-2),
               3  =>  ones(n-3),
               -1 => -ones(n-1))

    b  = randn(n)
    x0 = zeros(n)

    max_iter = 400
    restart  = 400
    tol      = 1e-10

    fmtstr(fmts) = join(["$(f):$(count(==(f), fmts))" for f in unique(fmts)], "  ")

    x,     errors,          true_errors    = restarted_gmres(A, b, x0, max_iter, restart, tol)
    println("Float64 only     : residual=$(round(norm(b.-A*x)/norm(b),    sigdigits=3))  iters=$(length(errors)-1)")

    x_hi, errors_hi, fmts_hi, true_errors_hi = relaxed_gmres(A, b, x0, max_iter, restart, tol, 1)
    println("Relaxed high→low : residual=$(round(norm(b.-A*x_hi)/norm(b), sigdigits=3))  iters=$(length(errors_hi)-1)  $(fmtstr(fmts_hi))")

    x_lo, errors_lo, fmts_lo, true_errors_lo = relaxed_gmres_low(A, b, x0, max_iter, restart, tol, 1)
    println("Relaxed low→high : residual=$(round(norm(b.-A*x_lo)/norm(b), sigdigits=3))  iters=$(length(errors_lo)-1)  $(fmtstr(fmts_lo))")

    # left: projected Krylov residuals (what GMRES minimises each step)
    p = plot(0:length(errors)-1,    errors    ./ norm(b),
             yscale=:log10, label="Float64 only", lw=2)
    plot!(p, 0:length(errors_hi)-1, errors_hi ./ norm(b), label="high→low", lw=2)
    plot!(p, 0:length(errors_lo)-1, errors_lo ./ norm(b), label="low→high",  lw=2)
    hline!(p, [4.88e-4], ls=:dot,  color=:grey,  label="Float16 floor")
    hline!(p, [tol],     ls=:dash, color=:black, label="tol")
    xlabel!(p, "Iteration"); ylabel!(p, "Projected Residual / ||b||")
    title!(p, "Projected Krylov Residual")

    # right: true residual at every inner step — same x-axis, same point count
    p2 = plot(0:length(true_errors)-1,    true_errors    ./ norm(b),
              yscale=:log10, label="Float64 only", lw=2)
    plot!(p2, 0:length(true_errors_hi)-1, true_errors_hi ./ norm(b), label="high→low", lw=2)
    plot!(p2, 0:length(true_errors_lo)-1, true_errors_lo ./ norm(b), label="low→high",  lw=2)
    hline!(p2, [4.88e-4], ls=:dot,  color=:grey,  label="Float16 floor")
    hline!(p2, [tol],     ls=:dash, color=:black, label="tol")
    xlabel!(p2, "Iteration"); ylabel!(p2, "True Residual / ||b||")
    title!(p2, "True Residual (||b - Ax|| / ||b||)")

    display(plot(p, p2, layout=(1, 2), size=(1200, 500),
                 plot_title="n=$n, restart=$restart, tol=$tol"))
end

