include("qr__orth_cgs.jl")
include("qr__orth_cgs2.jl")
include("qr__orth_dcgs2.jl")
include("qr__orth_hh_lvl1.jl")
include("qr__orth_hh_lvl2.jl")
include("qr__orth_mgs_lvl1.jl")
include("qr__orth_mgs_lvl2.jl")
include("qr__orth_stabilitymetric.jl")
using LinearAlgebra, Plots

let
    m = 500
    n = 100
    range_log10KA = 0:1:16

    condA        = zeros(length(range_log10KA))
    orth_cgs     = zeros(length(range_log10KA))
    orth_cgs2    = zeros(length(range_log10KA))
    orth_dcgs2   = zeros(length(range_log10KA))
    orth_hh1     = zeros(length(range_log10KA))
    orth_hh2     = zeros(length(range_log10KA))
    orth_mgs1    = zeros(length(range_log10KA))
    orth_mgs2    = zeros(length(range_log10KA))
    repres_cgs   = zeros(length(range_log10KA))
    repres_cgs2  = zeros(length(range_log10KA))
    repres_dcgs2 = zeros(length(range_log10KA))
    repres_hh1   = zeros(length(range_log10KA))
    repres_hh2   = zeros(length(range_log10KA))
    repres_mgs1  = zeros(length(range_log10KA))
    repres_mgs2  = zeros(length(range_log10KA))

    for (i, log10KA) in enumerate(range_log10KA)
        U, _ = qr(randn(m, n))
        V, _ = qr(randn(n, n))
        S = Diagonal(10 .^ LinRange(0, log10KA, n))
        A = Matrix(U) * Matrix(S) * Matrix(V)'
        condA[i] = cond(A)

        Q, R = qr__orth_cgs(A);     repres_cgs[i],   orth_cgs[i],   _ = qr__orth_stabilitymetric(A, Q, R)
        Q, R = qr__orth_cgs2(A);    repres_cgs2[i],  orth_cgs2[i],  _ = qr__orth_stabilitymetric(A, Q, R)
        Q, R = qr__orth_dcgs2(A);   repres_dcgs2[i], orth_dcgs2[i], _ = qr__orth_stabilitymetric(A, Q, R)
        Q, R = qr__orth_hh_lvl1(A); repres_hh1[i],   orth_hh1[i],   _ = qr__orth_stabilitymetric(A, Q, R)
        Q, R = qr__orth_hh_lvl2(A); repres_hh2[i],   orth_hh2[i],   _ = qr__orth_stabilitymetric(A, Q, R)
        Q, R = qr__orth_mgs_lvl1(A); repres_mgs1[i], orth_mgs1[i],  _ = qr__orth_stabilitymetric(A, Q, R)
        Q, R = qr__orth_mgs_lvl2(A); repres_mgs2[i], orth_mgs2[i],  _ = qr__orth_stabilitymetric(A, Q, R)
    end

    x = 10 .^ collect(range_log10KA)
    p = plot(x, orth_cgs,   xscale=:log10, yscale=:log10, label="CGS",      lw=2, marker=:ltriangle)
    plot!(p, x, orth_cgs2,  label="CGS2",      lw=2, marker=:ltriangle)
    plot!(p, x, orth_dcgs2, label="DCGS2",     lw=2, marker=:square)
    plot!(p, x, orth_hh1,   label="HH lvl1",   lw=2, marker=:utriangle)
    plot!(p, x, orth_hh2,   label="HH lvl2",   lw=2, marker=:square)
    plot!(p, x, orth_mgs1,  label="MGS lvl1",  lw=2, marker=:utriangle)
    plot!(p, x, orth_mgs2,  label="MGS lvl2",  lw=2, marker=:square)
    xlabel!(p, "Condition Number")
    ylabel!(p, "|| I - Q'Q ||_fro")
    title!(p, "Orthogonality")

    p2 = plot(x, repres_cgs,   xscale=:log10, yscale=:log10, label="CGS",      lw=2, marker=:ltriangle)
    plot!(p2, x, repres_cgs2,  label="CGS2",      lw=2, marker=:ltriangle)
    plot!(p2, x, repres_dcgs2, label="DCGS2",     lw=2, marker=:square)
    plot!(p2, x, repres_hh1,   label="HH lvl1",   lw=2, marker=:utriangle)
    plot!(p2, x, repres_hh2,   label="HH lvl2",   lw=2, marker=:square)
    plot!(p2, x, repres_mgs1,  label="MGS lvl1",  lw=2, marker=:utriangle)
    plot!(p2, x, repres_mgs2,  label="MGS lvl2",  lw=2, marker=:square)
    xlabel!(p2, "Condition Number")
    ylabel!(p2, "|| A - QR ||_fro / ||A||_fro")
    title!(p2, "Representation Error")

    display(plot(p, p2, layout=(1, 2), size=(1200, 500),
                 plot_title="QR Comparison, m=$m, n=$n"))
end
