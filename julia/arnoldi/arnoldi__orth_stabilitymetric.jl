using LinearAlgebra

function arnoldi__orth_stabilitymetric(A, b::AbstractVector,
                                        Q::AbstractMatrix, H::AbstractMatrix,
                                        beta)
    m = size(A, 1)
    k = size(H, 1)
    # orth methods return beta as a scalar or a 1-element vector
    β = isa(beta, AbstractVector) ? beta[1] : beta

    nrmA   = norm(reshape(A, :))
    repres = zeros(k)
    orth   = zeros(k)
    condn  = zeros(k)
    normS  = zeros(k)

    for j in 1:k
        Qj = Q[:, 1:j]
        orth[j]   = norm(reshape(Matrix(I, j, j) .- Qj' * Qj, :))
        repres[j] = norm(hcat(b, A * Q[:, 1:j-1]) .- Qj * hcat([β; zeros(j-1)], H[1:j, 1:j-1])) / nrmA
        condn[j]  = cond(hcat(b, A * Q[:, 1:j-1]))
        U = triu(Qj' * Qj, 1)
        S = (Matrix(I, j, j) .+ U) \ U
        normS[j] = opnorm(S, 2)
    end

    return repres, orth, condn, normS
end
