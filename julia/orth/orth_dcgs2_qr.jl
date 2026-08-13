using LinearAlgebra

function orth_dcgs2_qr(Q::AbstractMatrix, R::AbstractVector)
    m = size(Q, 1)
    j = size(Q, 2)
    r = zeros(j - 1, 2)
    q = zeros(m, 2)

    if j == 2
        r[1, 1] = Q[:, 1]' * Q[:, 1]
        r[1, 2] = Q[:, 1]' * Q[:, 2]
        r[1, 1] = sqrt(r[1, 1])
        r[1, 2] = r[1, 2] / r[1, 1]
        q[:, 1] = Q[:, 1] ./ r[1, 1]
        q[:, 2] = Q[:, 2] .- q[:, 1] .* r[1, 2]
    end

    if j >= 3
        tmp = Q[:, 1:j-1]' * Q[:, j-1:j]   # (j-1)×2
        r[1:j-2, 1] = tmp[1:j-2, 1]
        r[1:j-1, 2] = tmp[:, 2]
        r[j-1, 1]   = tmp[j-1, 1]

        r[j-1, 2] -= r[1:j-2, 1]' * r[1:j-2, 2]
        r[j-1, 1] -= r[1:j-2, 1]' * r[1:j-2, 1]
        r[1:j-2, 1] = R[1:j-2] .+ r[1:j-2, 1]
        r[j-1, 1] = sqrt(r[j-1, 1])
        r[j-1, 2] = r[j-1, 2] / r[j-1, 1]

        Qjm1 = Q[:, j-1] .- Q[:, 1:j-2] * tmp[1:j-2, 1]
        Qj   = Q[:, j]   .- Q[:, 1:j-2] * r[1:j-2, 2]
        q[:, 1] = Qjm1 ./ r[j-1, 1]
        q[:, 2] = Qj .- q[:, 1] .* r[j-1, 2]
    end

    return q, r
end
