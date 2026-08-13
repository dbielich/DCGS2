using LinearAlgebra

function orth_mgs_lvl1_Ainnerprod(M::AbstractMatrix, MQ::AbstractMatrix,
                                   Q::AbstractMatrix, a::AbstractVector)
    m = size(Q, 1)
    j = size(Q, 2) + 1
    r  = zeros(j)
    q  = zeros(m)
    mq = zeros(m)
    a  = copy(a)
    if j > 1
        for i in 1:j-1
            r[i] = a' * MQ[:, i]
            a .-= r[i] .* Q[:, i]
        end
    end
    mq .= M * a
    tmp = a' * mq
    r[j] = sign(tmp) * sqrt(abs(tmp))
    q  .= a  ./ r[j]
    mq .= mq ./ r[j]
    return mq, q, r
end
