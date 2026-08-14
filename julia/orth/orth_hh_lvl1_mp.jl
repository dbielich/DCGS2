using LinearAlgebra

# Mixed-precision Householder orthogonalization step.
#
# Four independent precision knobs (Tisseur et al. arXiv:2602.18134 inspiration):
#
#   T_work   — working/output precision.  Replaces all hardcoded Float64 so the
#              entire function can run at e.g. Float32 or BigFloat.  tau is kept
#              at T_work regardless of T_push/T_pop/T_construct.
#
#   T_push   — precision for applying the accumulated reflectors to the input
#              vector `a` (we "push" a into the subspace).  Inner products
#              V[:,i]'*a and the subtract accumulate at T_push.
#              NOTE: on x86 CPU, Float16 arithmetic is emulated in Float32 —
#              the bits are chopped at Float16 on store/load but intermediate
#              accumulation uses Float32 hardware.  True fp16 accumulation
#              requires GPU or ARM fp16 hardware (e.g. Apple Silicon).
#
#   T_pop    — precision for recovering the explicit basis vector q from the
#              accumulated reflectors (we "pop" back out of the subspace).
#              T_pop and T_push are typically the same but are separated for
#              experimentation.
#
#   T_construct — precision for computing normx/norma when forming the new
#               Householder reflector.  tau (t) is always kept at T_work
#               even when T_construct is lower, preserving accurate orthogonality.
#               Default: T_work (no precision reduction in the reflector step).
#
# The returned `a` (Householder vector) is at T_work; the caller casts it to
# whatever storage type is desired before writing into V.
function orth_hh_lvl1_mp(V::AbstractMatrix, tau::AbstractVector, a::AbstractVector;
                          T_work   ::Type = Float64,
                          T_push   ::Type = T_work,
                          T_pop    ::Type = T_work,
                          T_construct::Type = T_work)
    m = size(V, 1)
    j = size(V, 2) + 1
    r = zeros(T_work, j)
    q = zeros(T_work, m)
    a = T_work.(copy(a))

    # Push: apply accumulated reflectors to a at T_push precision.
    # Inner products and subtracts accumulate at T_push; result is promoted
    # back to T_work immediately after each step.
    for i in 1:j-1
        vi    = T_push.(V[i:m, i])
        alpha = T_work(tau[i] * (vi' * T_push.(a[i:m])))
        a[i:m] .-= T_work.(vi) .* alpha
    end

    if j > 1
        r[1:j-1] = a[1:j-1]
    end

    # Compute new Householder reflector at T_construct precision for normx/norma.
    # tau (t) is formed at T_work to preserve orthogonality regardless of T_construct.
    sub   = T_construct.(a[j:m])
    normx = T_work(norm(sub[2:end]))
    norma = sqrt(normx^2 + T_work(sub[1])^2)
    r[j]  = T_work(sub[1]) > zero(T_work) ? -norma : norma

    a[j]      = a[j] > zero(T_work) ? a[j] + norma : a[j] - norma
    a[j+1:m] ./= a[j]
    t         = T_work(2) / (one(T_work) + (normx / a[j])^2)   # tau at T_work
    a[1:j-1] .= zero(T_work)
    a[j]      = one(T_work)

    # Pop: recover explicit orthonormal vector q at T_pop precision.
    q[j]      = -t
    q[j+1:m]  = a[j+1:m] .* q[j]
    q[j]      = one(T_work) + q[j]
    for i in j-1:-1:1
        vi      = T_pop.(V[i+1:m, i])
        q[i]    = T_work(-tau[i] * (vi' * T_pop.(q[i+1:m])))
        q[i+1:m] .+= T_work.(vi) .* q[i]
    end

    return q, t, r, a   # a is T_work; caller casts to storage type
end
