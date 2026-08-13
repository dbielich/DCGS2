using LinearAlgebra

function orth_dcgs2_qr_cleanup(Q::AbstractMatrix, R::AbstractVector)
    m = size(Q, 1)
    n = size(Q, 2)
    r = zeros(n)

    r .= Q' * Q[:, n]
    r[n] -= r[1:n-1]' * r[1:n-1]
    r[n] = sqrt(r[n])

    q = (Q[:, n] .- Q[:, 1:n-1] * r[1:n-1]) ./ r[n]
    r[1:n-1] = R[1:n-1] .+ r[1:n-1]

    return q, r
end
