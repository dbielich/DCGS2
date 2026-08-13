using LinearAlgebra

function orth_mgs_lvl2_Ainnerprod_2sync(M::AbstractMatrix, MQ::AbstractMatrix,
                                         Q::AbstractMatrix, T::AbstractMatrix,
                                         a::AbstractVector)
    m = size(Q, 1)
    j = size(Q, 2) + 1
    r  = zeros(j)
    t  = zeros(j)
    q  = zeros(m)
    mq = zeros(m)
    a  = copy(a)
    if j > 1
        r[1:j-1] = T[1:j-1, 1:j-1]' * (MQ' * a)
        a .-= Q * r[1:j-1]
    end
    mq .= M * a
    tmp = a' * mq
    if j > 1
        t[1:j-1] = T[1:j-1, 1:j-1] * (-(MQ' * a))
        t[j] = 1.0
    end
    r[j] = sign(tmp) * sqrt(abs(tmp))
    q  .= a  ./ r[j]
    mq .= mq ./ r[j]
    t[1:j-1] ./= r[j]
    t[j] = 1.0
    return mq, q, t, r
end
