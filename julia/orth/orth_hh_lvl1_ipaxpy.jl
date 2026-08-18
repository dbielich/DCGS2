using LinearAlgebra

# Householder lvl1 with ip/axpy/norm precision knobs, matching the
# Gram-Schmidt MP interface so cross-method comparison is meaningful.
#
#   T_ip   — inner products in BOTH the push phase (v'a) and pop phase (v'q)
#   T_axpy — vector updates  in BOTH the push phase (a -= v*α) and pop phase (q += v*qi)
#   T_norm — normx / norma computation when constructing the new reflector
#
# Applying T_ip and T_axpy to both phases keeps Q (from pop) and R (from push)
# consistent, so ‖A-QR‖ and ‖I-Q'Q‖ degrade together as in CGS/MGS.
function orth_hh_lvl1_ipaxpy(V::AbstractMatrix, tau::AbstractVector, a::AbstractVector;
                               T_work::Type = Float64,
                               T_ip  ::Type = T_work,
                               T_axpy::Type = T_work,
                               T_norm::Type = T_work)
    m = size(V, 1)
    j = size(V, 2) + 1
    r = zeros(T_work, j)
    q = zeros(T_work, m)
    a = T_work.(copy(a))

    # Push phase: apply accumulated reflectors to a.
    for i in 1:j-1
        alpha = T_work(tau[i] * (T_ip.(V[i:m, i])' * T_ip.(a[i:m])))
        a[i:m] .-= T_work.(T_axpy.(V[i:m, i]) .* T_axpy(alpha))
    end

    if j > 1
        r[1:j-1] = a[1:j-1]
    end

    # Construct new reflector; use T_norm values consistently for norma and the
    # a[j] update so the reflector and the recorded r[j] stay in agreement.
    sub   = T_norm.(a[j:m])
    normx = T_work(norm(sub[2:end]))
    aj    = T_work(sub[1])
    norma = sqrt(normx^2 + aj^2)
    r[j]  = aj > zero(T_work) ? -norma : norma

    a[j]      = aj > zero(T_work) ? aj + norma : aj - norma
    a[j+1:m] ./= a[j]
    t         = T_work(2) / (one(T_work) + (normx / a[j])^2)
    a[1:j-1] .= zero(T_work)
    a[j]      = one(T_work)

    # Pop phase: recover explicit orthonormal vector q.
    q[j]     = -t
    q[j+1:m] = a[j+1:m] .* q[j]
    q[j]     = one(T_work) + q[j]
    for i in j-1:-1:1
        q[i] = T_work(-tau[i] * (T_ip.(V[i+1:m, i])' * T_ip.(q[i+1:m])))
        q[i+1:m] .+= T_work.(T_axpy.(V[i+1:m, i]) .* T_axpy(q[i]))
    end

    return q, t, r, a
end
