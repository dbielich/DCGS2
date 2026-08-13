using LinearAlgebra

function orth_mgs_lvl2_Ainnerprod_orth_projector(M::AbstractMatrix, MQ::AbstractMatrix,
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
        r[1:j-1] = MQ' * a
        r[1:j-1] = T[1:j-1, 1:j-1]' \ r[1:j-1]
        r[1:j-1] = T[1:j-1, 1:j-1]  \ r[1:j-1]
        a .-= Q * r[1:j-1]
    end
    mq .= M * a
    tmp = a' * mq
    r[j] = sign(tmp) * sqrt(abs(tmp))
    q  .= a  ./ r[j]
    mq .= mq ./ r[j]
    if j > 1
        t[1:j-1] = T[1:j-1, 1:j-1]' * (MQ' * q)
        t[j] = sqrt(1.0 - t[1:j-1]' * t[1:j-1])
    else
        t[j] = 1.0
    end
    return mq, q, t, r
end
