include("../orth/orth_hh_lvl1_mp.jl")
using LinearAlgebra

# Mixed-precision Arnoldi using orth_hh_lvl1_mp.
#
# T_store  — storage type for the Householder vectors in V (e.g. Float32).
#            Mimics Tisseur et al.: compute/store reflectors cheaply, apply accurately.
# T_apply  — precision for inner products when applying/recovering reflectors.
# T_construct — precision for computing each new Householder reflector.
# tau      — always Float64 regardless of other choices.
function arnoldi__orth_hh_lvl1_mp(A, b::AbstractVector, k::Int;
                                   T_work    ::Type = Float64,
                                   T_store   ::Type = T_work,
                                   T_push    ::Type = T_work,
                                   T_pop     ::Type = T_work,
                                   T_construct ::Type = T_work)
    m   = size(A, 1)
    Q   = zeros(Float64, m, k)
    V   = zeros(T_store,  m, k)   # reflectors stored at T_store
    H   = zeros(Float64,  k, k-1)
    tau = zeros(Float64,  k)       # tau at T_work (Float64 default)

    # Initial step with empty arrays — establishes first Householder.
    Q[:, 1], tau[1], beta_vec, v1 = orth_hh_lvl1_mp(
        zeros(T_store, m, 0), tau[1:0], b;
        T_work=T_work, T_push=T_push, T_pop=T_pop, T_construct=T_construct)
    V[:, 1] = T_store.(v1)

    for j in 2:k
        w = A * Q[:, j-1]   # matvec in Float64
        Q[:, j], tau[j], H[1:j, j-1], v_new = orth_hh_lvl1_mp(
            V[:, 1:j-1], tau[1:j-1], w;
            T_work=T_work, T_push=T_push, T_pop=T_pop, T_construct=T_construct)
        V[:, j] = T_store.(v_new)
    end

    return Q, H, beta_vec   # beta_vec[1] = ±norm(b)
end
