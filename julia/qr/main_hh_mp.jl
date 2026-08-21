include("../orth/orth_hh_lvl1_mp_test.jl")
include("qr__orth_stabilitymetric.jl")
using LinearAlgebra, Plots, Printf, Random, Measures
# For F16/BF16: add `using BFloat16s` and extend configs below.

Random.seed!(0)
m = 500; n = 100
range_log10KA = 0:1:16
npts = length(range_log10KA)
fn(T) = T == Float64 ? "F64" : T == Float32 ? "F32" : T == Float16 ? "F16" : "BF16"

# QR driver using orth_hh_lvl1_mp_test.
# Maintains V_push (T_store_push precision) and V_pop (T_store_pop precision)
# simultaneously. Both receive the same F64 reflector from construct, cast
# to their respective storage precision. tau is always F64.
function run_hh_dual(A;
                      T_store_push ::Type = Float64,
                      T_store_pop  ::Type = Float64,
                      T_push_arith ::Type = Float64,
                      T_pop_arith  ::Type = Float64)
    m, n  = size(A)
    V_push = zeros(T_store_push, m, n)
    V_pop  = zeros(T_store_pop,  m, n)
    tau    = zeros(Float64, n)        # single F64 tau (construct always F64)
    Q      = zeros(Float64, m, n)
    R      = zeros(Float64, n, n)

    for j in 1:n
        q, t, r, v_new = orth_hh_lvl1_mp_test(
            V_push[:, 1:j-1], tau[1:j-1],
            V_pop[:, 1:j-1],  tau[1:j-1],   # same tau for both phases
            A[:, j];
            T_push_arith=T_push_arith, T_pop_arith=T_pop_arith)

        V_push[:, j] = T_store_push.(v_new)  # store at push precision
        V_pop[:, j]  = T_store_pop.(v_new)   # store at pop precision
        tau[j]       = t
        Q[:, j]      = q
        R[1:j, j]    = r
    end
    return Q, R
end

# ── Config table ──────────────────────────────────────────────────────────────
# (label, T_store_push, T_store_pop, T_push_arith, T_pop_arith)
# Construct is always Float64. tau is always Float64.
# To extend to F16/BF16: add rows with Float16 / BFloat16 types.
configs = [
    # Pure baselines
    ("All F64",
        Float64, Float64, Float64, Float64),
    ("All F32",
        Float32, Float32, Float32, Float32),

    # F64 V both, but push arithmetic reduced
    ("V=F64                           arith=F32/F64",
        Float64, Float64, Float32, Float64),

    # Opposite: F64 push, F32 pop
    ("Vpush=F32 Vpop=F64  arith=F64/F32",
        Float32, Float64, Float64, Float32),

    # F32 stored push V, F64 arithmetic for push (F32 data, F64 compute)
    ("Vpush=F32 Vpop=F64  arith=F32/F64",
        Float32, Float64, Float32, Float64),

    # Core request: F32 push side, F64 pop side (storage AND arithmetic match)
    ("Vpush=F32 Vpop=F32  arith=F32/F64",
        Float32, Float32, Float32, Float64),

    # F32 stored pop V, F64 arithmetic for pop
    ("Vpush=F32 Vpop=F32  arith=F64/F64",
        Float32, Float32, Float64, Float64),
]

# ── Sweep ─────────────────────────────────────────────────────────────────────
orth_curves    = [zeros(npts) for _ in configs]
repres_curves  = [zeros(npts) for _ in configs]
orth_curves_32   = [zeros(npts) for _ in configs]
repres_curves_32 = [zeros(npts) for _ in configs]

println("="^72)
println("  HH lvl1 dual-V MP test: storage and arithmetic per phase")
println("  m=$m  n=$n  construct always Float64")
println("="^72)

for (k, log10KA) in enumerate(range_log10KA)
    U, _ = qr(randn(m, n)); Vr, _ = qr(randn(n, n))
    S = Diagonal(10.0 .^ LinRange(0, log10KA, n))
    A = Matrix(U) * Matrix(S) * Matrix(Vr)'

    for (c, (_, Tsp, Tsq, Tpa, Tqa)) in enumerate(configs)
        # F64 A run
        Q, R = run_hh_dual(A; T_store_push=Tsp, T_store_pop=Tsq,
                               T_push_arith=Tpa, T_pop_arith=Tqa)
        Q64 = Float64.(Q); R64 = Float64.(R); A64 = Float64.(A)
        orth_curves[c][k]   = norm(reshape(Matrix(I, n, n) .- Q64'*Q64, :))
        repres_curves[c][k] = norm(A64 .- Q64*R64) / norm(A64)

        # F32 A run — A[:,j] enters orth_hh_lvl1_mp_test as Float32, promoted to F64 inside
        Q32, R32 = run_hh_dual(Float32.(A); T_store_push=Tsp, T_store_pop=Tsq,
                                             T_push_arith=Tpa, T_pop_arith=Tqa)
        Q32f = Float64.(Q32); R32f = Float64.(R32)
        orth_curves_32[c][k]   = norm(reshape(Matrix(I, n, n) .- Q32f'*Q32f, :))
        repres_curves_32[c][k] = norm(A64 .- Q32f*R32f) / norm(A64)  # compare vs true F64 A
    end
end

for (c, (lbl, _, _, _, _)) in enumerate(configs)
    @printf("  %-40s  orth=%.2e  repr=%.2e\n",
            lbl, orth_curves[c][end], repres_curves[c][end])
end
println("="^72)

# ── Plot ──────────────────────────────────────────────────────────────────────
styles = [:solid, :solid, :dash, :dash, :dot, :dot, :dashdot, :dashdot]
colors = [:black, :red, :blue, :orange, :green, :purple, :cyan, :brown]

clamp_plot(v) = max.(min.(v, 1.0), eps(Float64))
plot_kw = (xscale=:log10, yscale=:log10, lw=1.5, legendfontsize=7, titlefontsize=9,
           tickfontsize=7, guidefontsize=8, framestyle=:box,
           bottom_margin=5mm, left_margin=3mm)

x_axis = 10.0 .^ collect(range_log10KA)
eps64  = eps(Float64); eps32 = eps(Float32)

p_orth   = plot(; title="[A=F64] Orthogonality  ‖I-Q'Q‖_fro", xlabel="κ(A)", ylabel="", plot_kw...)
p_repres = plot(; title="[A=F64] Representation  ‖A-QR‖/‖A‖",   xlabel="κ(A)", ylabel="", plot_kw...)
p_orth32   = plot(; title="[A=F32] Orthogonality  ‖I-Q'Q‖_fro", xlabel="κ(A)", ylabel="", plot_kw...)
p_repres32 = plot(; title="[A=F32] Representation  ‖A-QR‖/‖A‖",   xlabel="κ(A)", ylabel="", plot_kw...)

for (c, (lbl, _, _, _, _)) in enumerate(configs)
    plot!(p_orth,     x_axis, clamp_plot(orth_curves[c]),    label=lbl, ls=styles[c], color=colors[c])
    plot!(p_repres,   x_axis, clamp_plot(repres_curves[c]),  label=lbl, ls=styles[c], color=colors[c])
    plot!(p_orth32,   x_axis, clamp_plot(orth_curves_32[c]),   label=lbl, ls=styles[c], color=colors[c])
    plot!(p_repres32, x_axis, clamp_plot(repres_curves_32[c]), label=lbl, ls=styles[c], color=colors[c])
end
for p in (p_orth, p_orth32)
    plot!(p, x_axis, clamp_plot(x_axis .^ 2 .* eps64), ls=:dot, color=:grey,     label="κ²·F64eps", lw=1)
    plot!(p, x_axis, clamp_plot(x_axis .^ 2 .* eps32), ls=:dot, color=:lightblue, label="κ²·F32eps", lw=1)
end

sz = (1200, 520)
f1 = tempname() * "_hh_f64A_residuals.png"
f2 = tempname() * "_hh_f64A_orthogonality.png"
f3 = tempname() * "_hh_f32A_residuals.png"
f4 = tempname() * "_hh_f32A_orthogonality.png"
savefig(plot(p_orth,   p_repres;   layout=(1,2), size=sz, plot_title="A=Float64  m=$m n=$n"), f1)
savefig(plot(p_orth;   size=(750, 520), plot_title="A=Float64 — Orthogonality"), f2)
savefig(plot(p_orth32, p_repres32; layout=(1,2), size=sz, plot_title="A=Float32  m=$m n=$n (repr vs F64 A)"), f3)
savefig(plot(p_orth32; size=(750, 520), plot_title="A=Float32 — Orthogonality"), f4)
println("Saved: $f1"); println("Saved: $f2")
println("Saved: $f3"); println("Saved: $f4")
run(Cmd(vcat(["open"], [f1, f2, f3, f4])))
