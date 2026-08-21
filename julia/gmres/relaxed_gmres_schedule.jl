include("../orth/orth_hh_lvl1.jl")
using LinearAlgebra

# Schedule-based inexact GMRES.
#
# A precision schedule is a Vector of (thresh_floor, Type) pairs sorted in
# DESCENDING order of thresh_floor.  The format used at step j is the Type
# from the first pair whose thresh_floor ≤ eta * rel_res.
#
# Predefined schedules (mirroring relaxed_gmres and relaxed_gmres_low):
#   SCHEDULE_HIGH_LOW = [(4.88e-4, Float64), (5.96e-8, Float32), (-Inf, Float16)]
#   SCHEDULE_LOW_HIGH = [(4.88e-4, Float16), (5.96e-8, Float32), (-Inf, Float64)]
#
# Custom examples:
#   [(1.0, Float64), (1e-3, Float32), (-Inf, Float64)]   — Float32 only for mid-range
#   [(1.0, Float16), (1e-6, Float32), (-Inf, Float64)]   — aggressive low→high
#
# Any type T can appear in the schedule provided T.(array) and T arithmetic work.
# On Apple Silicon (M-series), Float16 is hardware-accelerated (native NEON FP16).
# BFloat16 is available via BFloat16s.jl and works the same way.
#
const SCHEDULE_HIGH_LOW = [(4.88e-4, Float64), (5.96e-8, Float32), (-Inf, Float16)]
const SCHEDULE_LOW_HIGH = [(4.88e-4, Float16), (5.96e-8, Float32), (-Inf, Float64)]

function select_format(eta, rel_res, schedule)
    thresh = eta * rel_res
    for (floor, T) in schedule
        thresh >= floor && return T
    end
    return Float64
end

function matvec_typed(A, q::AbstractVector, ::Type{T}) where T
    T === Float64 && return A * q, "Float64"
    return Float64.(T.(A) * T.(q)), string(T)
end

function relaxed_gmres_schedule(A, b::AbstractVector, x0::AbstractVector,
                                  max_iter::Int, restart::Int, tol::Real,
                                  eta::Real, schedule::Vector)
    n     = length(b)
    nrm_b = norm(b)
    x     = copy(x0)
    r     = b .- A * x
    beta  = norm(r)
    errors       = [beta]
    true_errors  = [beta]
    orth_history = [0.0]
    formats      = String[]

    if beta < tol * nrm_b
        return x, errors, true_errors, orth_history, formats
    end

    iter     = 0
    res_prev = beta

    while iter < max_iter
        m   = min(restart, max_iter - iter)
        Q   = zeros(Float64, n, m + 1)
        V   = zeros(Float64, n, m + 1)
        H   = zeros(Float64, m + 1, m)
        tau = zeros(Float64, m + 1)

        r    = b .- A * x
        beta = norm(r)
        if iter > 0
            push!(errors,       beta)
            push!(true_errors,  beta)
            push!(orth_history, 0.0)
        end
        Q[:, 1], tau[1], h_init, V[:, 1] = orth_hh_lvl1(V[:, 1:0], tau[1:0], r)

        g    = zeros(m + 1)
        g[1] = h_init[1]

        for j in 1:m
            iter += 1

            T   = select_format(eta, res_prev / nrm_b, schedule)
            w, fmt = matvec_typed(A, Q[:, j], T)
            push!(formats, fmt)

            Q[:, j+1], tau[j+1], H[1:j+1, j], V[:, j+1] = orth_hh_lvl1(V[:, 1:j], tau[1:j], w)

            H_sub    = H[1:j+1, 1:j]
            g_sub    = g[1:j+1]
            y        = H_sub \ g_sub
            res_norm = norm(H_sub * y .- g_sub)

            push!(errors,      res_norm)
            push!(true_errors, norm(b .- A * (x .+ Q[:, 1:j] * y)))
            push!(orth_history, norm(Matrix(I, j+1, j+1) - Q[:, 1:j+1]' * Q[:, 1:j+1]))
            res_prev = res_norm

            if res_norm < tol * nrm_b
                m = j; break
            end
        end

        y = H[1:m+1, 1:m] \ g[1:m+1]
        x .+= Q[:, 1:m] * y

        true_res = norm(b .- A * x)
        push!(true_errors, true_res)
        if true_res < tol * nrm_b; break; end
    end

    return x, errors, true_errors, orth_history, formats
end
