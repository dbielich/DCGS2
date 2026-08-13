include("../orth/orth_mgs_lvl1.jl")
using LinearAlgebra

function arnoldi__orth_mgs_lvl1(A, b::AbstractVector, k::Int)
    m = size(A, 1)
    Q = zeros(m, k)
    H = zeros(k, k - 1)

    beta = norm(b)
    Q[:, 1] = b ./ beta

    for j in 2:k
        Q[:, j] = A * Q[:, j-1]
        Q[:, j], H[1:j, j-1] = orth_mgs_lvl1(Q[:, 1:j-1], Q[:, j])
    end

    return Q, H, beta
end
