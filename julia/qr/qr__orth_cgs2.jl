include("../orth/orth_cgs2.jl")
using LinearAlgebra

function qr__orth_cgs2(A::AbstractMatrix)
    m, n = size(A)
    Q = zeros(m, n)
    R = zeros(n, n)
    for j in 1:n
        Q[:, j], R[1:j, j] = orth_cgs2(Q[:, 1:j-1], A[:, j])
    end
    return Q, R
end
