include("../orth/orth_hh_lvl1.jl")
using LinearAlgebra

function restarted_gmres(A, b::AbstractVector, x0::AbstractVector,
                          max_iter::Int, restart::Int, tol::Real)
    n = length(b)
    nrm_b = norm(b)
    x = copy(x0)
    r = b .- A * x
    beta = norm(r)
    errors      = [beta]
    true_errors = [beta]
    orth_history = [0.0]

    if beta < tol * nrm_b
        return x, errors, true_errors, orth_history
    end

    iter = 0
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

        for j in 1:m
            iter += 1
            w = A * Q[:, j]
            Q[:, j+1], tau[j+1], H[1:j+1, j], V[:, j+1] = orth_hh_lvl1(V[:, 1:j], tau[1:j], w)

            H_sub = H[1:j+1, 1:j]
            g_sub = g[1:j+1]
            y = H_sub \ g_sub

            res_norm = norm(H_sub * y .- g_sub)
            push!(errors, res_norm)
            push!(true_errors, norm(b .- A * (x .+ Q[:, 1:j] * y)))
            push!(orth_history, norm(Matrix(I, j+1, j+1) - Q[:, 1:j+1]' * Q[:, 1:j+1]))

            if res_norm < tol * nrm_b
                m = j
                break
            end
        end

        y = H[1:m+1, 1:m] \ g[1:m+1]
        x .+= Q[:, 1:m] * y

        true_res = norm(b .- A * x)
        push!(true_errors, true_res)
        if true_res < tol * nrm_b
            break
        end
    end

    return x, errors, true_errors, orth_history
end
