include("restarted_gmres.jl")
include("restarted_gmres_mp.jl")
include("relaxed_gmres.jl")
include("relaxed_gmres_low.jl")
include("relaxed_gmres_orth_mp.jl")
include("relaxed_gmres_schedule.jl")
using LinearAlgebra, SparseArrays, Plots, Printf, Random, Measures

Random.seed!(42)
n  = 60
A  = diagm(0  =>  ones(n),
       1  =>  ones(n-1),
       2  =>  ones(n-2),
       3  =>  ones(n-3),
       -1 => -ones(n-1))
b  = randn(n); x0 = zeros(n)

max_iter = 400; restart = 400; tol = 1e-10
nrm_b = norm(b)
fn(T) = T == Float64 ? "F64" : T == Float32 ? "F32" : "F16"

styles = [:solid, :dash, :dot, :dashdot, :solid, :dash, :dot, :dashdot, :solid, :dash]
colors = [:black, :blue, :red, :green, :cyan, :orange, :purple, :brown, :pink, :teal]

plot_kw = (yscale=:log10, lw=1.5, legendfontsize=7, titlefontsize=9,
           tickfontsize=7, guidefontsize=8, framestyle=:box,
           top_margin=2mm, bottom_margin=2mm, left_margin=3mm)

function row_plots(results, row_label)
    p1 = plot(; title="$row_label — projected", ylabel="residual/‖b‖",
                xlabel="iteration", plot_kw...)
    p2 = plot(; title="$row_label — true ‖b-Ax‖/‖b‖",
                ylabel="", xlabel="iteration", plot_kw...)
    for (i, (lbl, errs, terrs)) in enumerate(results)
        plot!(p1, 0:length(errs)-1,  errs  ./ nrm_b, label=lbl, ls=styles[i], color=colors[i])
        plot!(p2, 0:length(terrs)-1, terrs ./ nrm_b, label=lbl, ls=styles[i], color=colors[i])
    end
    hline!(p1, [tol], ls=:dot, color=:grey, label="tol", lw=1)
    hline!(p2, [tol], ls=:dot, color=:grey, label="tol", lw=1)
    return p1, p2
end

function orth_plot_row(orth_results, row_label)
    p = plot(; title="$row_label — ||I - Q'Q||_fro",
               ylabel="", xlabel="iteration", plot_kw...)
    for (i, (lbl, orth)) in enumerate(orth_results)
        plot!(p, 0:length(orth)-1, max.(orth, eps(Float64)), label=lbl,
              ls=styles[i], color=colors[i])
    end
    hline!(p, [eps(Float32)], ls=:dot, color=:blue, label="F32 eps", lw=1)
    hline!(p, [eps(Float16)], ls=:dot, color=:red,  label="F16 eps", lw=1)
    return p
end

# ── Row 1: MP Householder, F64 matvec ────────────────────────────────────────
mp_configs = [
    (Float64, Float64, Float64, Float64, Float64),
    (Float64, Float64, Float32, Float32, Float64),
    (Float64, Float64, Float16, Float16, Float64),
    (Float32, Float32, Float64, Float64, Float32),
    (Float32, Float32, Float32, Float32, Float32),
    (Float32, Float32, Float16, Float16, Float32),
    (Float64, Float64, Float64, Float32, Float64),
    (Float64, Float64, Float32, Float64, Float64),
]
mp_lbl(Ts, Tw, Tpu, Tpo, Tc) = "s=$(fn(Ts)) push=$(fn(Tpu)) pop=$(fn(Tpo))"

println("\n── Row 1: MP Householder orth  (F64 matvec, n=$n, restart=$restart) ───────")
mp_res = []; mp_orth = []
for (Ts, Tw, Tpu, Tpo, Tc) in mp_configs
    _, errs, terrs, orth_e = restarted_gmres_mp(A, b, x0, max_iter, restart, tol;
                                                  T_store=Ts, T_work=Tw,
                                                  T_push=Tpu, T_pop=Tpo, T_construct=Tc)
    res = terrs[end] / nrm_b
    lab = mp_lbl(Ts, Tw, Tpu, Tpo, Tc)
    @printf("  %-34s  %.2e  %4d  %s\n", lab, res, length(errs)-1, res < tol ? "✓" : "✗")
    push!(mp_res,  (lab, errs, terrs))
    push!(mp_orth, (lab, orth_e))
end

# ── Row 2: Relaxed matvec, F64 Householder (schedule-based) ──────────────────
rlx_schedules = [
    ("F64 (base)",             [(-Inf, Float64)]),
    ("high→low [F64,F32,F16]", SCHEDULE_HIGH_LOW),
    ("low→high [F16,F32,F64]", SCHEDULE_LOW_HIGH),
    ("mid-only [F64,F32]",     [(4.88e-4, Float64), (-Inf, Float32)]),
    ("F32 always",             [(-Inf, Float32)]),
]

println("\n── Row 2: Relaxed matvec  (F64 Householder, eta=1.0, n=$n, restart=$restart) ─")
rlx_res = []; rlx_orth = []
for (lbl, sched) in rlx_schedules
    _, errs, terrs, orth_e, _ = relaxed_gmres_schedule(A, b, x0, max_iter, restart, tol, 1.0, sched)
    res = terrs[end] / nrm_b
    @printf("  %-30s  %.2e  %4d  %s\n", lbl, res, length(errs)-1, res < tol ? "✓" : "✗")
    push!(rlx_res,  (lbl, errs, terrs))
    push!(rlx_orth, (lbl, orth_e))
end

# ── Row 3: Combined relaxed matvec + MP Householder ──────────────────────────
# All hi→lo matvec schedule. lo→hi-unique F32 store/work variants ported here.
# Tuple: (sched_vec, mv_symbol, eta, T_store, T_work, T_push, T_pop, T_construct)
comb_configs = [
    ([(-Inf,Float64)],  :high2low, 1.0, Float64, Float64, Float64, Float64, Float64),  # baseline
    (SCHEDULE_HIGH_LOW, :high2low, 1.0, Float64, Float64, Float64, Float64, Float64),  # hi→lo, all F64
    (SCHEDULE_HIGH_LOW, :high2low, 1.0, Float64, Float64, Float32, Float32, Float64),  # hi→lo, F64 store, F32 push/pop
    (SCHEDULE_HIGH_LOW, :high2low, 1.0, Float64, Float64, Float64, Float32, Float64),  # hi→lo, F64 store, F64 push F32 pop
    (SCHEDULE_HIGH_LOW, :high2low, 1.0, Float32, Float32, Float64, Float64, Float32),  # hi→lo, F32 store, F64 push/pop
    (SCHEDULE_HIGH_LOW, :high2low, 1.0, Float32, Float32, Float32, Float32, Float32),  # hi→lo, F32 store, F32 push/pop
    (SCHEDULE_HIGH_LOW, :high2low, 1.0, Float32, Float32, Float32, Float64, Float32),  # hi→lo, F32 store, F32 push F64 pop
    (SCHEDULE_HIGH_LOW, :high2low, 1.0, Float32, Float32, Float64, Float32, Float32),  # hi→lo, F32 store, F64 push F32 pop
]
comb_lbl(sched, Ts, Tpu, Tpo) =
    "$(sched===SCHEDULE_HIGH_LOW ? "hi→lo" : "F64") s=$(fn(Ts)) push=$(fn(Tpu)) pop=$(fn(Tpo))"

println("\n── Row 3: Combined relaxed matvec + MP Householder  (n=$n, restart=$restart) ─")
comb_res = []; comb_orth = []
for (sched, mv_sym, eta, Ts, Tw, Tpu, Tpo, Tc) in comb_configs
    lab = comb_lbl(sched, Ts, Tpu, Tpo)
    _, errs, terrs, orth_e, _ = relaxed_gmres_orth_mp(A, b, x0, max_iter, restart, tol, eta;
                                                        matvec_schedule=mv_sym,
                                                        T_store=Ts, T_work=Tw,
                                                        T_push=Tpu, T_pop=Tpo, T_construct=Tc)
    res = terrs[end] / nrm_b
    @printf("  %-40s  %.2e  %4d  %s\n", lab, res, length(errs)-1, res < tol ? "✓" : "✗")
    push!(comb_res,  (lab, errs, terrs))
    push!(comb_orth, (lab, orth_e))
end

# ── Build plots ───────────────────────────────────────────────────────────────
p1, p2 = row_plots(mp_res,   "Row1 MP Householder")
p3, p4 = row_plots(rlx_res,  "Row2 Relaxed matvec")
p5, p6 = row_plots(comb_res, "Row3 Combined")

po1 = orth_plot_row(mp_orth,   "Row1 MP Householder")
po2 = orth_plot_row(rlx_orth,  "Row2 Relaxed matvec")
po3 = orth_plot_row(comb_orth, "Row3 Combined")

for p in (p1, p2, p3, p4, p5, p6, po1, po2, po3); plot!(p; bottom_margin=5mm); end

sz = (1000, 480)
f_r1  = tempname() * "_row1_mp_householder.png"
f_r2  = tempname() * "_row2_relaxed_matvec.png"
f_r3  = tempname() * "_row3_combined.png"
f_orth = tempname() * "_orthogonality.png"

savefig(plot(p1, p2;   layout=(1,2), size=sz, plot_title="Row 1 — MP Householder  n=$n restart=$restart"), f_r1)
savefig(plot(p3, p4;   layout=(1,2), size=sz, plot_title="Row 2 — Relaxed matvec  n=$n restart=$restart"), f_r2)
savefig(plot(p5, p6;   layout=(1,2), size=sz, plot_title="Row 3 — Combined        n=$n restart=$restart"), f_r3)
savefig(plot(po1, po2, po3; layout=(1,3), size=(1500,480), plot_title="Orthogonality comparison"), f_orth)

println("\nSaved: $f_r1"); println("Saved: $f_r2")
println("Saved: $f_r3"); println("Saved: $f_orth")
run(`open $f_r1 $f_r2 $f_r3 $f_orth`)
