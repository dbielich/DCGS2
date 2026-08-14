include("../orth/orth_hh_lvl1.jl")
using LinearAlgebra

# Inexact restarted GMRES: LOW precision early, HIGHER precision as convergence proceeds.
# eta: scales the threshold per step as thresh = eta * rel_res.
#   Large rel_res (early) → large budget → Float16 or Float32 matvec.
#   Small rel_res (late)  → small budget → Float64 matvec near convergence.
# This is the alternative precision schedule to relaxed_gmres.jl.
# Theory: Simoncini & Szyld (2003), SIAM J. Sci. Comput. 25(2), 454-477.
function relaxed_gmres_low(A, b::AbstractVector, x0::AbstractVector,
                             max_iter::Int, restart::Int, tol::Real, eta::Real = 1.0)
    n = length(b)
    nrm_b = norm(b)
    x = copy(x0)
    r = b .- A * x
    beta = norm(r)
    errors      = [beta]
    true_errors = [beta]
    formats     = String[]
    orth_history = [0.0]

    if beta < tol * nrm_b
        return x, errors, formats, true_errors, orth_history
    end

    iter = 0
    res_prev = beta

    while iter < max_iter
        m = min(restart, max_iter - iter)
        Q   = zeros(n, m + 1)
        V   = zeros(n, m + 1)
        H   = zeros(m + 1, m)
        tau = zeros(m + 1)

        r = b .- A * x
        beta = norm(r)
        if iter > 0
            push!(errors,       beta)
            push!(true_errors,  beta)
            push!(orth_history, 0.0)
        end
        Q[:, 1], tau[1], h_init, V[:, 1] = orth_hh_lvl1(V[:, 1:0], tau[1:0], r)

        g = zeros(m + 1)
        g[1] = h_init[1]

        precision_restart = false
        for j in 1:m
            iter += 1

            thresh_j = eta * (res_prev / nrm_b)
            w, fmt   = matvec_relaxed_low(A, Q[:, j], eta, res_prev / nrm_b)
            push!(formats, fmt)

            Q[:, j+1], tau[j+1], H[1:j+1, j], V[:, j+1] = orth_hh_lvl1(V[:, 1:j], tau[1:j], w)

            H_sub = H[1:j+1, 1:j]
            g_sub = g[1:j+1]
            y = H_sub \ g_sub

            res_norm = norm(H_sub * y .- g_sub)
            push!(errors,      res_norm)
            push!(true_errors, norm(b .- A * (x .+ Q[:, 1:j] * y)))
            push!(orth_history, norm(Matrix(I, j+1, j+1) - Q[:, 1:j+1]' * Q[:, 1:j+1]))
            res_prev = res_norm

            if res_norm < tol * nrm_b
                m = j; break
            end

            # Precision-floor restart: when low format stalls, advance to next higher format.
            fmt_floor = thresh_j >= 4.88e-4 ? 4.88e-4 :
                        thresh_j >= 5.96e-8 ? 5.96e-8 : 0.0
            if fmt_floor > 0.0 && res_norm < fmt_floor * nrm_b
                x .+= Q[:, 1:j] * y
                push!(true_errors, norm(b .- A * x))
                res_prev = (fmt_floor / (2 * eta)) * nrm_b  # shift thresh into next higher format
                precision_restart = true; m = j; break
            end
        end

        if !precision_restart
            y = H[1:m+1, 1:m] \ g[1:m+1]
            x .+= Q[:, 1:m] * y
            true_res = norm(b .- A * x)
            push!(true_errors, true_res)
            if true_res < tol * nrm_b; break; end
        else
            if true_errors[end] < tol * nrm_b; break; end
        end
    end

    return x, errors, formats, true_errors, orth_history
end

function matvec_relaxed_low(A, q::AbstractVector, eta::Real, rel_res::Real)
    # thresh = eta * rel_res: large when residual is high, small when nearly converged.
    # Float16 u≈4.88e-4, Float32 u≈5.96e-8, Float64 u≈1.11e-16
    thresh = eta * rel_res
    if thresh >= 4.88e-4
        return Float64.(Float16.(A) * Float16.(q)), "Float16"
    elseif thresh >= 5.96e-8
        return Float64.(Float32.(A) * Float32.(q)), "Float32"
    else
        return A * q, "Float64"
    end
end
