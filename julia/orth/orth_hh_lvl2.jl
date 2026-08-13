using LinearAlgebra

function orth_hh_lvl2(V::AbstractMatrix, T::AbstractMatrix, a::AbstractVector)
    m = size(V, 1)
    j = size(V, 2) + 1
    q = zeros(m)
    v = zeros(m)
    r = zeros(j)
    t = zeros(j)
    a = copy(a)

    if j > 1
        r[1:j-1] = T[1:j-1, 1:j-1]' * (V' * a)
        a .-= V * r[1:j-1]
        r[1:j-1] = a[1:j-1]
    end

    normx = norm(a[j+1:m])
    norma = sqrt(normx^2 + a[j]^2)
    v[j] = a[j] > 0.0 ? a[j] + norma : a[j] - norma
    v[j+1:m] = a[j+1:m] ./ v[j]
    r[j] = a[j] > 0.0 ? -norma : norma
    tau = 2.0 / (1.0 + (normx / v[j])^2)
    v[j] = 1.0

    q[j] = -tau
    q[j+1:m] = v[j+1:m] .* q[j]
    q[j] = 1.0 + q[j]
    if j > 1
        t[1:j-1] = T[1:j-1, 1:j-1] * (V' * q)
        q .-= V * t[1:j-1]
    end

    if j > 1
        t[1:j-1] = -tau .* (V' * v)
        t[1:j-1] = T[1:j-1, 1:j-1] * t[1:j-1]
    end
    t[j] = tau

    return q, t, r, v
end
