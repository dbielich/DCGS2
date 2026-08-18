using LinearAlgebra

# Mixed-precision CGS2 orthogonalization step (two-pass reorthogonalization).
# Same precision knobs as orth_cgs_mp: T_ip, T_axpy, T_norm.
# Both passes use the same precision settings.
function orth_cgs2_mp(Q::AbstractMatrix, a::AbstractVector;
                       T_work::Type = Float64,
                       T_ip  ::Type = T_work,
                       T_axpy::Type = T_work,
                       T_norm::Type = T_work)
    m = size(Q, 1)
    j = size(Q, 2) + 1
    r = zeros(T_work, j)
    h = zeros(T_work, j)
    q = T_work.(copy(a))

    if j > 1
        # First pass
        r[1:j-1] = T_work.(T_ip.(Q)' * T_ip.(q))
        q .-= T_work.(T_axpy.(Q) * T_axpy.(r[1:j-1]))

        # Second pass (reorthogonalization)
        h[1:j-1] = T_work.(T_ip.(Q)' * T_ip.(q))
        q .-= T_work.(T_axpy.(Q) * T_axpy.(h[1:j-1]))

        r[1:j-1] .+= h[1:j-1]
    end

    r[j] = T_work(norm(T_norm.(q)))
    q ./= r[j]

    return q, r
end
