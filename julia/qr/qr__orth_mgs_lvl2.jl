include("../orth/orth_mgs_lvl2.jl")
using LinearAlgebra

function qr__orth_mgs_lvl2(A::AbstractMatrix)
    m, n = size(A)
    Q = zeros(m, n)
    R = zeros(n, n)
    T = zeros(n, n)
    for j in 1:n
        Q[:, j], T[1:j, j], R[1:j, j] = orth_mgs_lvl2(Q[:, 1:j-1], T[1:j-1, 1:j-1], A[:, j])
    end
    return Q, R
end
