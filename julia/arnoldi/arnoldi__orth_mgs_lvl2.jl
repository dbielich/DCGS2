include("../orth/orth_mgs_lvl2.jl")
using LinearAlgebra

function arnoldi__orth_mgs_lvl2(A, b::AbstractVector, k::Int)
    m = size(A, 1)
    Q = zeros(m, k)
    H = zeros(k, k - 1)
    T = zeros(k, k)

    beta = norm(b)
    Q[:, 1] = b ./ beta
    T[1, 1] = 1.0

    for j in 2:k
        Q[:, j] = A * Q[:, j-1]
        Q[:, j], T[1:j, j], H[1:j, j-1] = orth_mgs_lvl2(Q[:, 1:j-1], T[1:j-1, 1:j-1], Q[:, j])
    end

    return Q, H, beta
end
