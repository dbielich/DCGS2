include("qr__orth_mgs_lvl1_Ainnerprod.jl")
include("qr__orth_mgs_lvl2_Ainnerprod.jl")
include("qr__orth_mgs_lvl2_Ainnerprod_2sync.jl")
include("qr__orth_mgs_lvl2_Ainnerprod_orth_projector.jl")
include("qr__orth_mgs_lvl2_Ainnerprod_orth_projector_v2.jl")
include("qr_Ainnerprod_orth_stabilitymetric.jl")
using LinearAlgebra, Plots

m = 100
n = 50
M = rand(m, m)
M = 0.5 * (M + M')
M = M + m * Matrix(I, m, m)

range_log10KA = 0:1:16
condA              = zeros(length(range_log10KA))
orth_mgs1          = zeros(length(range_log10KA))
orth_mgs2          = zeros(length(range_log10KA))
orth_mgs2_2sync    = zeros(length(range_log10KA))
orth_mgs2_opv1     = zeros(length(range_log10KA))
orth_mgs2_opv2     = zeros(length(range_log10KA))
repres_mgs1        = zeros(length(range_log10KA))
repres_mgs2        = zeros(length(range_log10KA))
repres_mgs2_2sync  = zeros(length(range_log10KA))
repres_mgs2_opv1   = zeros(length(range_log10KA))
repres_mgs2_opv2   = zeros(length(range_log10KA))

for (i, log10KA) in enumerate(range_log10KA)
    U, _ = qr(randn(m, n))
    V, _ = qr(randn(n, n))
    S = Diagonal(10 .^ LinRange(0, log10KA, n))
    A = Matrix(U) * Matrix(S) * Matrix(V)'
    condA[i] = cond(A)

    MQ, Q, R = qr__orth_mgs_lvl1_Ainnerprod(A, M);                    repres_mgs1[i],       orth_mgs1[i],       _ = qr_Ainnerprod_orth_stabilitymetric(A, M, Q, R)
    MQ, Q, R = qr__orth_mgs_lvl2_Ainnerprod(A, M);                    repres_mgs2[i],       orth_mgs2[i],       _ = qr_Ainnerprod_orth_stabilitymetric(A, M, Q, R)
    MQ, Q, R = qr__orth_mgs_lvl2_Ainnerprod_2sync(A, M);              repres_mgs2_2sync[i], orth_mgs2_2sync[i], _ = qr_Ainnerprod_orth_stabilitymetric(A, M, Q, R)
    MQ, Q, R = qr__orth_mgs_lvl2_Ainnerprod_orth_projector(A, M);     repres_mgs2_opv1[i],  orth_mgs2_opv1[i],  _ = qr_Ainnerprod_orth_stabilitymetric(A, M, Q, R)
    MQ, Q, R = qr__orth_mgs_lvl2_Ainnerprod_orth_projector_v2(A, M);  repres_mgs2_opv2[i],  orth_mgs2_opv2[i],  _ = qr_Ainnerprod_orth_stabilitymetric(A, M, Q, R)
end

x = 10 .^ collect(range_log10KA)
p = plot(x, orth_mgs1,       xscale=:log10, yscale=:log10, label="MGS lvl1",       lw=2, marker=:utriangle)
plot!(p, x, orth_mgs2,       label="MGS lvl2",       lw=2, marker=:square)
plot!(p, x, orth_mgs2_2sync, label="MGS lvl2 2sync", lw=2, marker=:diamond)
plot!(p, x, orth_mgs2_opv1,  label="MGS lvl2 OP v1", lw=2, marker=:circle)
plot!(p, x, orth_mgs2_opv2,  label="MGS lvl2 OP v2", lw=2, marker=:star5)
xlabel!(p, "Condition Number of Input Matrix")
ylabel!(p, "M-Orthogonality || I - Q'MQ ||_fro")
title!(p, "A-Inner Product Orthogonality, QR factorization\nm = $m, n = $n")
display(p)
