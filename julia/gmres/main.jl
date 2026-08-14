include("restarted_gmres.jl")
include("relaxed_gmres.jl")
include("relaxed_gmres_low.jl")
include("relaxed_gmres_schedule.jl")
using LinearAlgebra, SparseArrays, Plots, Random

# count occurrences of each format without StatsBase
countmap(v) = Dict(k => count(==(k), v) for k in unique(v))

demo_nilpotent = false  # bidiagonal nilpotent: too easy for GMRES(m), restarts recover cheaply
demo_grcar     = false  # Grcar matrix: reliable stagnation due to pseudospectrum surrounding origin
demo_schedule  = true  # custom precision schedule array comparison

if demo_schedule
    # Custom precision schedule: define an array of (thresh_floor, Type) pairs.
    # thresh = eta * rel_res; first pair where thresh >= floor determines the format.
    # Add BFloat16 via: using BFloat16s; then include (1e-2, BFloat16) in a schedule.
    Random.seed!(42)
    n  = 60
    A  = diagm(0  =>  ones(n),
           1  =>  ones(n-1),
           2  =>  ones(n-2),
           3  =>  ones(n-3),
           -1 => -ones(n-1))
    b  = randn(n); x0 = zeros(n)
    max_iter = 400; restart = 400; tol = 1e-10; eta = 1.0

    schedules = [
        ("high→low  [F64,F32,F16]", SCHEDULE_HIGH_LOW),
        ("low→high  [F16,F32,F64]", SCHEDULE_LOW_HIGH),
        ("mid-only  [F64,F32]", [(4.88e-4, Float64), (-Inf, Float32)]),
        ("F32 always",              [(-Inf, Float32)]),
        ("F64 always",              [(-Inf, Float64)]),
    ]

    println("\n── Custom schedule comparison ─────────────────────────────────────────")
    println("  n=$n  restart=$restart  tol=$tol  eta=$eta")
    println("  Schedule                     residual   iters  formats")
    println("  " * "─"^60)
    styles = [:solid, :dash, :dot, :dashdot, :solid, :dash, :dot, :dashdot]
    colors = [:black, :blue, :red, :green, :cyan, :orange, :purple, :brown]
    nrm_b  = norm(b)

    p  = plot(; yscale=:log10, title="Projected Krylov Residual",
               xlabel="Iteration", ylabel="Residual / ||b||", legend=:topright)
    p2 = plot(; yscale=:log10, title="True Residual  ||b - Ax|| / ||b||",
               xlabel="Iteration", ylabel="",               legend=:topright)

    for (i, (lbl, sched)) in enumerate(schedules)
        _, errs, terrs, _, fmts = relaxed_gmres_schedule(A, b, x0, max_iter, restart, tol, eta, sched)
        res = terrs[end] / nrm_b
        @printf("  %-28s  %.2e  %4d   %s\n", lbl, res, length(terrs)-1,
                join(["$(f):$(count(==(f),fmts))" for f in unique(fmts)], " "))
        plot!(p,  0:length(errs)-1,  errs  ./ nrm_b, label=lbl, lw=2, ls=styles[i], color=colors[i])
        plot!(p2, 0:length(terrs)-1, terrs ./ nrm_b, label=lbl, lw=2, ls=styles[i], color=colors[i])
    end
    hline!(p,  [4.88e-4], ls=:dot, color=:grey,  label="Float16 floor", lw=1)
    hline!(p,  [tol],     ls=:dash, color=:black, label="tol",           lw=1)
    hline!(p2, [4.88e-4], ls=:dot, color=:grey,  label="Float16 floor", lw=1)
    hline!(p2, [tol],     ls=:dash, color=:black, label="tol",           lw=1)
    display(plot(p, p2, layout=(1, 2), size=(1200, 500),
                 plot_title="Custom schedules  |  n=$n, restart=$restart, tol=$tol"))

elseif demo_grcar
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

    _, err_full, _, _  = restarted_gmres(A, b, x0, n*3,  n, tol)
    _, err_r5, _, _    = restarted_gmres(A, b, x0, n*30, 5, tol)
    _, err_r15, _, _   = restarted_gmres(A, b, x0, n*10, 15, tol)

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

    _, err_full, _, _      = restarted_gmres(A, b, x0, n,    n,  tol)
    _, err_restarted, _, _ = restarted_gmres(A, b, x0, n*5, 10,  tol)

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
    A  = diagm(0  =>  ones(n),
               1  =>  ones(n-1),
               2  =>  ones(n-2),
               3  =>  ones(n-3),
               -1 => -ones(n-1))
    b  = randn(n); x0 = zeros(n)

    max_iter = 400; restart = 400; tol = 1e-10; eta = 1.0
    nrm_b = norm(b)

    fmtstr(fmts) = join(["$(f):$(count(==(f), fmts))" for f in unique(fmts)], "  ")

    x, errors, true_errors, _ = restarted_gmres(A, b, x0, max_iter, restart, tol)
    println("Float64 only     : n=$n restart=$restart  residual=$(round(norm(b.-A*x)/nrm_b, sigdigits=3))  iters=$(length(errors)-1)")

    _, errs_hi, terrs_hi, _, fmts_hi = relaxed_gmres_schedule(
        A, b, x0, max_iter, restart, tol, eta, SCHEDULE_HIGH_LOW)
    println("high→low  sched  : n=$n restart=$restart  residual=$(round(terrs_hi[end]/nrm_b, sigdigits=3))  iters=$(length(errs_hi)-1)  $(fmtstr(fmts_hi))")

    _, errs_lo, terrs_lo, _, fmts_lo = relaxed_gmres_schedule(
        A, b, x0, max_iter, restart, tol, eta, SCHEDULE_LOW_HIGH)
    println("low→high  sched  : n=$n restart=$restart  residual=$(round(terrs_lo[end]/nrm_b, sigdigits=3))  iters=$(length(errs_lo)-1)  $(fmtstr(fmts_lo))")

    p = plot(0:length(errors)-1,   errors   ./ nrm_b,
             yscale=:log10, label="Float64 only", lw=2)
    plot!(p, 0:length(errs_hi)-1, errs_hi ./ nrm_b, label="high→low", lw=2)
    plot!(p, 0:length(errs_lo)-1, errs_lo ./ nrm_b, label="low→high", lw=2)
    hline!(p, [4.88e-4], ls=:dot, color=:grey,  label="Float16 floor")
    hline!(p, [tol],     ls=:dash, color=:black, label="tol")
    xlabel!(p, "Iteration"); ylabel!(p, "Projected Residual / ||b||")
    title!(p, "Projected Krylov Residual")

    p2 = plot(0:length(true_errors)-1,  true_errors ./ nrm_b,
              yscale=:log10, label="Float64 only", lw=2)
    plot!(p2, 0:length(terrs_hi)-1, terrs_hi ./ nrm_b, label="high→low", lw=2)
    plot!(p2, 0:length(terrs_lo)-1, terrs_lo ./ nrm_b, label="low→high", lw=2)
    hline!(p2, [4.88e-4], ls=:dot,  color=:grey,  label="Float16 floor")
    hline!(p2, [tol],     ls=:dash, color=:black, label="tol")
    xlabel!(p2, "Iteration"); ylabel!(p2, "True Residual / ||b||")
    title!(p2, "True Residual  ||b - Ax|| / ||b||")

    display(plot(p, p2, layout=(1, 2), size=(1200, 500),
                 plot_title="n=$n, restart=$restart, tol=$tol  (schedule-based)"))
end

