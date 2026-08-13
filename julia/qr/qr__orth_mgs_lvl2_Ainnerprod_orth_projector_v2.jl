include("../orth/orth_mgs_lvl2_Ainnerprod_orth_projector_v2.jl")
using LinearAlgebra

function qr__orth_mgs_lvl2_Ainnerprod_orth_projector_v2(A::AbstractMatrix, M::AbstractMatrix)
    m, n = size(A)
    Q  = zeros(m, n)
    R  = zeros(n, n)
    MQ = zeros(m, n)
    T  = zeros(n, n)
    for j in 1:n
        MQ[:, j], Q[:, j], T[1:j, j], R[1:j, j] = orth_mgs_lvl2_Ainnerprod_orth_projector_v2(M, MQ[:, 1:j-1], Q[:, 1:j-1], T[1:j-1, 1:j-1], A[:, j])
    end
    return MQ, Q, R
end
