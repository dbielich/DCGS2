using LinearAlgebra

function orth_dcgs2_arnoldi(Q::AbstractMatrix, H::AbstractMatrix)
    m = size(Q, 1)
    j = size(Q, 2)
    h = zeros(j - 1)
    r = zeros(j - 1)
    q = zeros(m, 2)

    if j == 2
        r[1] = Q[:, 1]' * Q[:, 2]
        q[:, 1] = Q[:, 1]
        q[:, 2] = Q[:, 2] .- Q[:, 1:1] .* r[1]
    end

    if j > 2
        work = Q[:, 1:j-1]' * Q[:, j-1:j]   # (j-1)×2
        h[j-1] = sqrt(work[j-1, 1] - work[1:j-2, 1]' * work[1:j-2, 1])
        work[j-1, 2] = (work[j-1, 2] - work[1:j-2, 1]' * work[1:j-2, 2]) / h[j-1]^2
        work[1:j-2, 2] ./= h[j-1]

        q[:, 1] = Q[:, j-1] .- Q[:, 1:j-2] * work[1:j-2, 1]
        q[:, 1] ./= h[j-1]
        q[:, 2] = (Q[:, j] ./ h[j-1]) .- hcat(Q[:, 1:j-2], q[:, 1:1]) * work[:, 2]

        h[1:j-2] = H[1:j-2, j-2] .+ work[1:j-2, 1]
        r[1:j-1] = work[:, 2] .- (H[1:j-1, 1:j-2] * work[1:j-2, 1]) ./ h[j-1]
    end

    return q, r, h
end
