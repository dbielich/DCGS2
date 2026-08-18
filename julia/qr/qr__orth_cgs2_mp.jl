include("../orth/orth_cgs2_mp.jl")
using LinearAlgebra

function qr__orth_cgs2_mp(A::AbstractMatrix;
                            T_work::Type = Float64, T_store::Type = T_work,
                            T_ip  ::Type = T_work,  T_axpy::Type = T_work,
                            T_norm::Type = T_work)
    m, n = size(A)
    Q = zeros(T_store, m, n)
    R = zeros(T_work, n, n)
    for j in 1:n
        q, r = orth_cgs2_mp(Q[:, 1:j-1], A[:, j];
                             T_work=T_work, T_ip=T_ip, T_axpy=T_axpy, T_norm=T_norm)
        Q[:, j] = T_store.(q)
        R[1:j, j] = r
    end
    return Q, R
end
