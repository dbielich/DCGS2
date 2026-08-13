include("../orth/orth_hh_lvl1.jl")
using LinearAlgebra

function arnoldi__orth_hh_lvl1(A, b::AbstractVector, k::Int)
    m = size(A, 1)
    Q   = zeros(m, k)
    V   = zeros(m, k)
    H   = zeros(k, k - 1)
    tau = zeros(k)

    Q[:, 1], tau[1], beta, V[:, 1] = orth_hh_lvl1(V[:, 1:0], tau[1:0], b)

    for j in 2:k
        V[:, j] = A * Q[:, j-1]
        Q[:, j], tau[j], H[1:j, j-1], V[:, j] = orth_hh_lvl1(V[:, 1:j-1], tau[1:j-1], V[:, j])
    end

    return Q, H, beta
end
