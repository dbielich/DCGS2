using LinearAlgebra

# Mixed-precision MGS level-1 orthogonalization step.
# Three precision knobs matching the three main operations per loop iteration:
#   T_ip   — inner product:  r[i] = Q[:,i]' * a
#   T_axpy — vector update:  a   -= r[i] * Q[:,i]
#   T_norm — normalization:  r[j] = norm(a), q = a / r[j]
# In MGS the inner product and axpy are interleaved (unlike CGS which batches them).
function orth_mgs_lvl1_mp(Q::AbstractMatrix, a::AbstractVector;
                            T_work::Type = Float64,
                            T_ip  ::Type = T_work,
                            T_axpy::Type = T_work,
                            T_norm::Type = T_work)
    m = size(Q, 1)
    j = size(Q, 2) + 1
    r = zeros(T_work, j)
    a = T_work.(copy(a))

    for i in 1:j-1
        # Inner product at T_ip, promoted back to T_work immediately
        r[i] = T_work(T_ip.(Q[:, i])' * T_ip.(a))
        # AXPY at T_axpy
        a .-= T_work.(T_axpy.(Q[:, i]) .* T_axpy(r[i]))
    end

    # Normalization at T_norm
    r[j] = T_work(norm(T_norm.(a)))
    q    = a ./ r[j]

    return q, r
end
