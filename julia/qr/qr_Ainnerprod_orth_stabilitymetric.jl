using LinearAlgebra

function qr_Ainnerprod_orth_stabilitymetric(A::AbstractMatrix, M::AbstractMatrix,
                                             Q::AbstractMatrix, R::AbstractMatrix)
    m, n = size(A)
    nrmA  = norm(reshape(A, :))
    orth  = norm(reshape(Matrix(I, n, n) .- Q' * M * Q, :))
    repres = norm(A .- Q * R) / nrmA
    U = triu(Q' * M * Q, 1)
    S = (Matrix(I, n, n) .+ U) \ U
    normS = opnorm(S, 2)
    return repres, orth, normS
end
