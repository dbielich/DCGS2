using LinearAlgebra

function qr__orth_stabilitymetric(A::AbstractMatrix, Q::AbstractMatrix, R::AbstractMatrix)
    m, n = size(A)
    nrmA  = norm(reshape(A, :))
    orth  = norm(reshape(Matrix(I, n, n) .- Q' * Q, :))
    repres = norm(A .- Q * R) / nrmA
    U = triu(Q' * Q, 1)
    S = (Matrix(I, n, n) .+ U) \ U
    normS = opnorm(S, 2)
    return repres, orth, normS
end
