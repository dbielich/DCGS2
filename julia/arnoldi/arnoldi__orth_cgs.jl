include("../orth/orth_cgs.jl")
using LinearAlgebra

function arnoldi__orth_cgs(A, b::AbstractVector, k::Int)
    m = size(A, 1)
    Q = zeros(m, k)
    H = zeros(k, k - 1)

    Q[:, 1], beta = orth_cgs(Q[:, 1:0], b)

    for j in 2:k
        Q[:, j] = A * Q[:, j-1]
        Q[:, j], H[1:j, j-1] = orth_cgs(Q[:, 1:j-1], Q[:, j])
    end

    return Q, H, beta
end
