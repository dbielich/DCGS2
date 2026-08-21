include("relaxed_gmres_schedule.jl")   # provides select_format, matvec_typed, SCHEDULE_*
include("../orth/orth_hh_lvl1_mp.jl")
using LinearAlgebra

# GMRES where BOTH the matvec AND the Householder orthogonalization (push/pop)
# follow adaptive precision schedules driven by the current relative residual.
#
# At each Arnoldi step j:
#   T_mv   = select_format(eta, rel_res, mv_schedule)          — matvec format
#   T_push = select_format(eta, rel_res, orth_push_schedule)   — push (apply reflectors to a)
#   T_pop  = select_format(eta, rel_res, orth_pop_schedule)    — pop  (recover q)
#
# orth_push_schedule and orth_pop_schedule default to mv_schedule (fully coupled).
# T_store and T_construct are fixed for the whole solve.
function relaxed_gmres_adaptive_orth(A, b::AbstractVector, x0::AbstractVector,
                                      max_iter::Int, restart::Int, tol::Real, eta::Real;
                                      mv_schedule         = SCHEDULE_HIGH_LOW,
                                      orth_push_schedule  = mv_schedule,
                                      orth_pop_schedule   = orth_push_schedule,
                                      T_store    ::Type   = Float64,
                                      T_work     ::Type   = Float64,
                                      T_construct::Type   = Float64)
    n = length(b)
    nrm_b = norm(b)
    x = copy(x0)
    r = b .- A * x
    beta = norm(r)
    errors       = [beta]
    true_errors  = [beta]
    orth_history = [0.0]
    mv_fmts  = String[]
    pu_fmts  = String[]   # push format per step (pop tracks same way via orth_pop_schedule)

    _fn(T) = T === Float64 ? "F64" : T === Float32 ? "F32" : "F16"

    if beta < tol * nrm_b
        return x, errors, true_errors, orth_history, mv_fmts, pu_fmts
    end

    iter = 0
    res_prev = beta

    while iter < max_iter
        m   = min(restart, max_iter - iter)
        Q   = zeros(Float64, n, m + 1)
        V   = zeros(T_store, n, m + 1)
        H   = zeros(Float64, m + 1, m)
        tau = zeros(Float64, m + 1)

        r    = b .- A * x
        beta = norm(r)
        if iter > 0
            push!(errors,       beta)
            push!(true_errors,  beta)
            push!(orth_history, 0.0)
        end
        Q[:, 1], tau[1], h_init, v1 = orth_hh_lvl1_mp(
            zeros(T_store, n, 0), tau[1:0], r;
            T_work=T_work, T_push=T_work, T_pop=T_work, T_construct=T_construct)
        V[:, 1] = T_store.(v1)

        g    = zeros(m + 1)
        g[1] = h_init[1]

        for j in 1:m
            iter    += 1
            rel_res  = res_prev / nrm_b

            T_mv   = select_format(eta, rel_res, mv_schedule)
            T_push = select_format(eta, rel_res, orth_push_schedule)
            T_pop  = select_format(eta, rel_res, orth_pop_schedule)
            print("iter=$iter, j=$j, rel_res=$(round(rel_res, sigdigits=3)), T_mv=$(_fn(T_mv)), T_push=$(_fn(T_push)), T_pop=$(_fn(T_pop))\n")

            w, _ = matvec_typed(A, Q[:, j], T_mv)
            push!(mv_fmts, _fn(T_mv))
            push!(pu_fmts, _fn(T_push))

            Q[:, j+1], tau[j+1], H[1:j+1, j], v_new = orth_hh_lvl1_mp(
                V[:, 1:j], tau[1:j], w;
                T_work=T_work, T_push=T_push, T_pop=T_pop, T_construct=T_construct)
            V[:, j+1] = T_store.(v_new)

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

    return x, errors, true_errors, orth_history, mv_fmts, pu_fmts
end
