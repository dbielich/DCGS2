include("../orth/orth_hh_lvl1.jl")
using LinearAlgebra

function qr__orth_hh_lvl1(A::AbstractMatrix)
    m, n = size(A)
    Q   = zeros(m, n)
    V   = zeros(m, n)
    R   = zeros(n, n)
    tau = zeros(n)
    for j in 1:n
        Q[:, j], tau[j], R[1:j, j], V[:, j] = orth_hh_lvl1(V[:, 1:j-1], tau[1:j-1], A[:, j])
    end
    return Q, R
end
