using LinearAlgebra

function orth_mgs_lvl2(Q::AbstractMatrix, T::AbstractMatrix, a::AbstractVector)
    m = size(Q, 1)
    j = size(Q, 2) + 1
    r = zeros(j)
    t = zeros(j)
    q = zeros(m)
    if j == 1
        q .= a
        t[1] = 1.0
    else
        r[1:j-1] = T[1:j-1, 1:j-1]' * (Q' * a)
        q .= a .- Q * r[1:j-1]
    end
    r[j] = norm(q)
    q ./= r[j]
    if j > 1
        t[1:j-1] = T[1:j-1, 1:j-1] * (-(Q' * q))
        t[j] = 1.0
    end
    return q, t, r
end
