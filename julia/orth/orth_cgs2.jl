using LinearAlgebra

function orth_cgs2(Q::AbstractMatrix, a::AbstractVector)
    m = size(Q, 1)
    j = size(Q, 2) + 1
    r = zeros(j)
    h = zeros(j)
    q = zeros(m)
    if j == 1
        q .= a
    else
        r[1:j-1] = Q' * a
        q .= a .- Q * r[1:j-1]
        h[1:j-1] = Q' * q
        q .-= Q * h[1:j-1]
        r[1:j-1] .+= h[1:j-1]
    end
    r[j] = norm(q)
    q ./= r[j]
    return q, r
end
