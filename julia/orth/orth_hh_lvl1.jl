using LinearAlgebra

function orth_hh_lvl1(V::AbstractMatrix, tau::AbstractVector, a::AbstractVector)
    m = size(V, 1)
    j = size(V, 2) + 1
    r = zeros(j)
    q = zeros(m)
    a = copy(a)

    for i in 1:j-1
        alpha = tau[i] * (V[i:m, i]' * a[i:m])
        a[i:m] .-= V[i:m, i] .* alpha
    end

    if j > 1
        r[1:j-1] = a[1:j-1]
    end

    normx = norm(a[j+1:m])
    norma = sqrt(normx^2 + a[j]^2)
    r[j] = a[j] > 0.0 ? -norma : norma
    a[j] = a[j] > 0.0 ? a[j] + norma : a[j] - norma
    a[j+1:m] ./= a[j]
    t = 2.0 / (1.0 + (normx / a[j])^2)
    a[1:j-1] .= 0.0
    a[j] = 1.0

    q[j] = -t
    q[j+1:m] = a[j+1:m] .* q[j]
    q[j] = 1.0 + q[j]
    for i in j-1:-1:1
        q[i] = -tau[i] * (V[i+1:m, i]' * q[i+1:m])
        q[i+1:m] .+= V[i+1:m, i] .* q[i]
    end

    return q, t, r, a
end
