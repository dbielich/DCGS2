using LinearAlgebra

# Mixed-precision Householder orthogonalization step.
#
# Precision roles (mirroring Tisseur et al. arXiv:2602.18134, Alg. 3):
#   T_apply  — precision for applying accumulated reflectors and recovering q.
#              Inner products  V[:,i]'*a  and  V[:,i]'*q  are computed at T_apply,
#              then the result is promoted back to Float64.
#   T_reflect — precision for computing the new Householder reflector
#               (normx, norma, and the updated subvector of a).
#   tau       — always Float64: the scalar factor is cheap to keep accurate and
#               critical for orthogonality (a low-precision tau breaks the reflection).
#
# The returned `a` is Float64; the caller decides the storage precision of V.
# This lets the caller experiment with:
#   V as Float32 → T_apply=Float64  (Tisseur: cheap storage, accurate application)
#   V as Float64 → T_apply=Float32  (accurate storage, cheap application)
#   V as Float32 → T_apply=Float32  (full low-precision Householder)
function orth_hh_lvl1_mp(V::AbstractMatrix, tau::AbstractVector, a::AbstractVector;
                          T_apply::Type  = Float64,
                          T_reflect::Type = Float64)
    m = size(V, 1)
    j = size(V, 2) + 1
    r = zeros(Float64, j)
    q = zeros(Float64, m)
    a = Float64.(copy(a))

    # Apply accumulated Householder reflectors at T_apply precision.
    # Promoting stored V columns to T_apply before the inner product then
    # immediately promoting the result back to Float64 mirrors the Tisseur
    # "promote-then-apply" strategy.
    for i in 1:j-1
        vi    = T_apply.(V[i:m, i])
        alpha = Float64(tau[i] * (vi' * T_apply.(a[i:m])))
        a[i:m] .-= Float64.(vi) .* alpha
    end

    if j > 1
        r[1:j-1] = a[1:j-1]
    end

    # Compute new Householder reflector at T_reflect precision.
    sub   = T_reflect.(a[j:m])
    normx = Float64(norm(sub[2:end]))
    norma = sqrt(normx^2 + Float64(sub[1])^2)
    r[j]  = Float64(sub[1]) > 0.0 ? -norma : norma

    # Build the Householder vector in Float64 (returned to caller for storage).
    a[j]       = a[j] > 0.0 ? a[j] + norma : a[j] - norma
    a[j+1:m] ./= a[j]
    t          = 2.0 / (1.0 + (normx / a[j])^2)  # tau — always Float64
    a[1:j-1]  .= 0.0
    a[j]       = 1.0

    # Recover explicit orthonormal vector q at T_apply precision.
    q[j]       = -t
    q[j+1:m]   = a[j+1:m] .* q[j]
    q[j]       = 1.0 + q[j]
    for i in j-1:-1:1
        vi      = T_apply.(V[i+1:m, i])
        q[i]    = Float64(-tau[i] * (vi' * T_apply.(q[i+1:m])))
        q[i+1:m] .+= Float64.(vi) .* q[i]
    end

    return q, t, r, a   # a is Float64; caller casts to desired storage type
end
