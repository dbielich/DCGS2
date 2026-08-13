include("../orth/orth_dcgs2_qr.jl")
include("../orth/orth_dcgs2_qr_cleanup.jl")
using LinearAlgebra

function qr__orth_dcgs2(A::AbstractMatrix)
    m, n = size(A)
    Q = copy(A)
    R = zeros(n, n)
    for j in 2:n
        q2, r2 = orth_dcgs2_qr(Q[:, 1:j], R[1:j-2, j-1])
        Q[:, j-1:j] = q2
        R[1:j-1, j-1:j] = r2
    end
    Q[:, n], R[1:n, n] = orth_dcgs2_qr_cleanup(Q[:, 1:n], R[1:n-1, n])
    return Q, R
end
