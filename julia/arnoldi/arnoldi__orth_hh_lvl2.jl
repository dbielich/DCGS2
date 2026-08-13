include("../orth/orth_hh_lvl2.jl")
using LinearAlgebra

function arnoldi__orth_hh_lvl2(A, b::AbstractVector, k::Int)
    m = size(A, 1)
    Q = zeros(m, k)
    V = zeros(m, k)
    H = zeros(k, k - 1)
    T = zeros(k, k)

    Q[:, 1], T[1:1, 1], beta, V[:, 1] = orth_hh_lvl2(V[:, 1:0], T[1:0, 1:0], b)

    for j in 2:k
        V[:, j] = A * Q[:, j-1]
        Q[:, j], T[1:j, j], H[1:j, j-1], V[:, j] = orth_hh_lvl2(V[:, 1:j-1], T[1:j-1, 1:j-1], V[:, j])
    end

    return Q, H, beta
end
