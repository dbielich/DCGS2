using LinearAlgebra

# Dual-V Householder orth step: push and pop each have their own V matrix
# (and matching tau), so you can independently control the precision of the
# reflectors seen by each phase within a single solve.
#
#   V_push / tau_push — Householder data used by the push phase
#   V_pop  / tau_pop  — Householder data used by the pop phase
#   T_push_arith      — arithmetic precision for the push inner products / axpy
#   T_pop_arith       — arithmetic precision for the pop inner products / axpy
#   Construct         — always Float64; caller stores the returned reflector at
#                       T_store_push in V_push and T_store_pop in V_pop.
#
# tau is always Float64 (construct is F64 regardless of V precision).
function orth_hh_lvl1_mp_test(
        V_push::AbstractMatrix, tau_push::AbstractVector,
        V_pop ::AbstractMatrix, tau_pop ::AbstractVector,
        a     ::AbstractVector;
        T_push_arith::Type = Float64,
        T_pop_arith ::Type = Float64)

    m = size(V_push, 1)
    j = size(V_push, 2) + 1
    r = zeros(Float64, j)
    q = zeros(Float64, m)
    a = Float64.(copy(a))   # working copy always F64

    # Push: apply P_1 … P_{j-1} using V_push at T_push_arith arithmetic.
    for i in 1:j-1
        vi    = T_push_arith.(V_push[i:m, i])
        alpha = Float64(tau_push[i] * (vi' * T_push_arith.(a[i:m])))
        a[i:m] .-= Float64.(vi) .* alpha
    end

    if j > 1; r[1:j-1] = a[1:j-1]; end

    # Construct: always Float64 — produces the new reflector vector and tau.
    normx  = norm(a[j+1:m])
    norma  = sqrt(normx^2 + a[j]^2)
    r[j]   = a[j] > 0.0 ? -norma : norma
    a[j]   = a[j] > 0.0 ? a[j] + norma : a[j] - norma
    a[j+1:m] ./= a[j]
    t      = 2.0 / (1.0 + (normx / a[j])^2)
    a[1:j-1] .= 0.0
    a[j]   = 1.0

    # Pop: recover q using V_pop at T_pop_arith arithmetic.
    q[j]     = -t
    q[j+1:m] = a[j+1:m] .* q[j]
    q[j]     = 1.0 + q[j]
    for i in j-1:-1:1
        vi      = T_pop_arith.(V_pop[i+1:m, i])
        q[i]    = Float64(-tau_pop[i] * (vi' * T_pop_arith.(q[i+1:m])))
        q[i+1:m] .+= Float64.(vi) .* q[i]
    end

    # a is the F64 reflector — caller stores at T_store_push / T_store_pop
    return q, t, r, a
end
