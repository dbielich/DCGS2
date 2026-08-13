include("arnoldi__orth_cgs.jl")
include("arnoldi__orth_cgs2.jl")
include("arnoldi__orth_dcgs2.jl")
include("arnoldi__orth_hh_lvl1.jl")
include("arnoldi__orth_hh_lvl2.jl")
include("arnoldi__orth_mgs_lvl1.jl")
include("arnoldi__orth_mgs_lvl2.jl")
include("arnoldi__orth_stabilitymetric.jl")
using LinearAlgebra, SparseArrays, Plots

let
    m = 100
    k = 80
    A = Diagonal(LinRange(1, m, m))
    A = Matrix(A)
    A[1, 1] = 1e-8
    b = randn(m)
    A = sparse(A)

    Q, H, beta = arnoldi__orth_cgs(A, b, k);      repres_cgs,  orth_cgs,  cond_cgs,  _ = arnoldi__orth_stabilitymetric(A, b, Q, H, beta)
    Q, H, beta = arnoldi__orth_cgs2(A, b, k);     repres_cgs2, orth_cgs2, cond_cgs2, _ = arnoldi__orth_stabilitymetric(A, b, Q, H, beta)
    Q, H, beta = arnoldi__orth_dcgs2(A, b, k);    repres_dcgs2,  orth_dcgs2,  cond_dcgs2,  _ = arnoldi__orth_stabilitymetric(A, b, Q, H, beta)
    Q, H, beta = arnoldi__orth_hh_lvl1(A, b, k);  repres_hh1,    orth_hh1,    cond_hh1,    _ = arnoldi__orth_stabilitymetric(A, b, Q, H, beta)
    Q, H, beta = arnoldi__orth_hh_lvl2(A, b, k);  repres_hh2,    orth_hh2,    cond_hh2,    _ = arnoldi__orth_stabilitymetric(A, b, Q, H, beta)
    Q, H, beta = arnoldi__orth_mgs_lvl1(A, b, k); repres_mgs1,   orth_mgs1,   cond_mgs1,   _ = arnoldi__orth_stabilitymetric(A, b, Q, H, beta)
    Q, H, beta = arnoldi__orth_mgs_lvl2(A, b, k); repres_mgs2,   orth_mgs2,   cond_mgs2,   _ = arnoldi__orth_stabilitymetric(A, b, Q, H, beta)

    x = 1:k
    p = plot(x, orth_cgs,   yscale=:log10, label="CGS",      lw=2, marker=:star5)
    plot!(p, x, orth_cgs2,  label="CGS2",     lw=2, marker=:star5)
    plot!(p, x, orth_dcgs2, label="DCGS2",    lw=2, marker=:square)
    plot!(p, x, orth_hh1,   label="HH lvl1",  lw=2, marker=:utriangle)
    plot!(p, x, orth_hh2,   label="HH lvl2",  lw=2, marker=:square)
    plot!(p, x, orth_mgs1,  label="MGS lvl1", lw=2, marker=:utriangle)
    plot!(p, x, orth_mgs2,  label="MGS lvl2", lw=2, marker=:square)
    xlabel!(p, "Iteration Count")
    ylabel!(p, "|| I - Q'Q ||_fro")
    title!(p, "Orthogonality")

    p2 = plot(x, repres_cgs,   yscale=:log10, label="CGS",      lw=2, marker=:star5)
    plot!(p2, x, repres_cgs2,  label="CGS2",     lw=2, marker=:star5)
    plot!(p2, x, repres_dcgs2, label="DCGS2",    lw=2, marker=:square)
    plot!(p2, x, repres_hh1,   label="HH lvl1",  lw=2, marker=:utriangle)
    plot!(p2, x, repres_hh2,   label="HH lvl2",  lw=2, marker=:square)
    plot!(p2, x, repres_mgs1,  label="MGS lvl1", lw=2, marker=:utriangle)
    plot!(p2, x, repres_mgs2,  label="MGS lvl2", lw=2, marker=:square)
    xlabel!(p2, "Iteration Count")
    ylabel!(p2, "|| [b,AQ] - Q[β,H] ||_fro / ||A||_fro")
    title!(p2, "Representation Error")

    display(plot(p, p2, layout=(1, 2), size=(1200, 500),
                 plot_title="Arnoldi Comparison, m=$m, k=$k"))
end
