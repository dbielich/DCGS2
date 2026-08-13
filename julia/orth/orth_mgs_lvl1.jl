using LinearAlgebra

function orth_mgs_lvl1(Q::AbstractMatrix, a::AbstractVector)
    m = size(Q, 1)
    j = size(Q, 2) + 1
    r = zeros(j)
    a = copy(a)
    if j > 1
        for i in 1:j-1
            r[i] = Q[:, i]' * a
            a .-= Q[:, i] .* r[i]
        end
    end
    r[j] = norm(a)
    q = a ./ r[j]
    return q, r
end
