include("../orth/orth_mgs_lvl1_Ainnerprod.jl")
using LinearAlgebra

function qr__orth_mgs_lvl1_Ainnerprod(A::AbstractMatrix, M::AbstractMatrix)
    m, n = size(A)
    Q  = zeros(m, n)
    R  = zeros(n, n)
    MQ = zeros(m, n)
    for j in 1:n
        MQ[:, j], Q[:, j], R[1:j, j] = orth_mgs_lvl1_Ainnerprod(M, MQ[:, 1:j-1], Q[:, 1:j-1], A[:, j])
    end
    return MQ, Q, R
end
