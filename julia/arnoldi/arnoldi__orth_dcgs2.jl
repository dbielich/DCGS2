include("../orth/orth_dcgs2_arnoldi.jl")
include("../orth/orth_dcgs2_arnoldi_cleanup.jl")
using LinearAlgebra

function arnoldi__orth_dcgs2(A, b::AbstractVector, k::Int)
    m = size(A, 1)
    Q = zeros(m, k)
    H = zeros(k, k - 1)

    beta = norm(b)
    Q[:, 1] = b ./ beta

    for j in 2:k
        Q[:, j] = A * Q[:, j-1]

        if j == 2
            q2, H[1:j-1, j-1], _ = orth_dcgs2_arnoldi(Q[:, 1:j], H[:, 1:0])
            Q[:, j-1:j] = q2
        else
            q2, H[1:j-1, j-1], H[1:j-1, j-2] = orth_dcgs2_arnoldi(Q[:, 1:j], H[:, 1:j-2])
            Q[:, j-1:j] = q2
        end
    end

    Q[:, k], H[1:k, k-1] = orth_dcgs2_arnoldi_cleanup(Q[:, 1:k], H[:, 1:k-1])

    return Q, H, beta
end
