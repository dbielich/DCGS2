using LinearAlgebra

# Mixed-precision CGS orthogonalization step.
# Three independent precision knobs matching the three main operations:
#   T_ip   — precision for inner products  (Q' * a, projection coefficients)
#   T_axpy — precision for vector updates  (a -= Q * r, applying projections)
#   T_norm — precision for normalization   (norm(q), q /= r[j])
# T_work is the working/output precision; T_ip/T_axpy/T_norm default to T_work.
function orth_cgs_mp(Q::AbstractMatrix, a::AbstractVector;
                      T_work::Type = Float64,
                      T_ip  ::Type = T_work,
                      T_axpy::Type = T_work,
                      T_norm::Type = T_work)
    m = size(Q, 1)
    j = size(Q, 2) + 1
    r = zeros(T_work, j)
    q = T_work.(copy(a))

    if j > 1
        # Inner products at T_ip, result promoted to T_work
        r[1:j-1] = T_work.(T_ip.(Q)' * T_ip.(q))
        # AXPY at T_axpy: subtract all projections at once
        q .-= T_work.(T_axpy.(Q) * T_axpy.(r[1:j-1]))
    end

    # Normalization at T_norm
    r[j] = T_work(norm(T_norm.(q)))
    q ./= r[j]

    return q, r
end
