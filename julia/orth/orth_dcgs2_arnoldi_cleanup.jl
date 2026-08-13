using LinearAlgebra

function orth_dcgs2_arnoldi_cleanup(Q::AbstractMatrix, H::AbstractMatrix)
    m = size(Q, 1)
    k = size(Q, 2)
    work = zeros(k)
    h = zeros(k)

    work[1:k-1] = Q[:, 1:k-1]' * Q[:, k]
    work[k]     = Q[:, k]' * Q[:, k]
    h[k] = sqrt(work[k] - work[1:k-1]' * work[1:k-1])
    q = (Q[:, k] .- Q[:, 1:k-1] * work[1:k-1]) ./ h[k]
    h[1:k-1] = H[1:k-1, k-1] .+ work[1:k-1]

    return q, h
end
