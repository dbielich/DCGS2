include("../orth/orth_hh_lvl1_mp.jl")
using LinearAlgebra

# T_store: precision for V (Householder reflectors). When push reads V[:,i],
# it reads a T_store value; T_push.(V[:,i]) then casts to the compute precision.
# T_pop=F64, T_construct=F64 are fixed so Q and R are built at full precision.
function qr__orth_hh_lvl1_mp(A::AbstractMatrix;
                               T_work    ::Type = Float64,
                               T_store   ::Type = Float64,
                               T_push    ::Type = Float64,
                               T_pop     ::Type = Float64,
                               T_construct::Type = Float64)
    m, n = size(A)
    Q   = zeros(Float64, m, n)
    V   = zeros(T_store,  m, n)   # reflectors stored at T_store
    R   = zeros(Float64, n, n)
    tau = zeros(Float64, n)
    for j in 1:n
        Q[:, j], tau[j], R[1:j, j], v_new = orth_hh_lvl1_mp(
            V[:, 1:j-1], tau[1:j-1], A[:, j];
            T_work=T_work, T_push=T_push, T_pop=T_pop, T_construct=T_construct)
        V[:, j] = T_store.(v_new)
    end
    return Q, R
end
