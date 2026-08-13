include("../orth/orth_mgs_lvl1.jl")
using LinearAlgebra

function qr__orth_mgs_lvl1(A::AbstractMatrix)
    m, n = size(A)
    Q = zeros(m, n)
    R = zeros(n, n)
    for j in 1:n
        Q[:, j], R[1:j, j] = orth_mgs_lvl1(Q[:, 1:j-1], A[:, j])
    end
    return Q, R
end
