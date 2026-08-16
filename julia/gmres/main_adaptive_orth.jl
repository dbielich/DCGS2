include("restarted_gmres.jl")
include("relaxed_gmres_schedule.jl")
include("relaxed_gmres_adaptive_orth.jl")
using LinearAlgebra, Plots, Printf, Random, Measures

Random.seed!(42)
n  = 60
F  = qr(randn(n, n)); Qm = Matrix(F.Q)
S  = Diagonal(10 .^ LinRange(0, 2, n))
A  = Qm * Matrix(S) * Qm'
A  = diagm(0  =>  ones(n),
       1  =>  ones(n-1),
       2  =>  ones(n-2),
       3  =>  ones(n-3),
       -1 => -ones(n-1))
b  = randn(n); x0 = zeros(n)
max_iter = 400; restart = 400; tol = 1e-10; eta = 1.0
nrm_b = norm(b)

F64 = [(-Inf, Float64)]   # fixed Float64 (no relaxation)

# Configs: (label, mv_schedule, orth_push_schedule, orth_pop_schedule)
configs = [
    ("F64 baseline",                      F64,               F64,               F64),
    ("hi→lo matvec  / F64 orth",          SCHEDULE_HIGH_LOW, F64,               F64),
    ("F64 matvec    / hi→lo orth",        F64,               SCHEDULE_HIGH_LOW, SCHEDULE_HIGH_LOW),
    ("hi→lo BOTH    (same schedule)",     SCHEDULE_HIGH_LOW, SCHEDULE_HIGH_LOW, SCHEDULE_HIGH_LOW),
    ("lo→hi matvec  / hi→lo orth",        SCHEDULE_LOW_HIGH, SCHEDULE_HIGH_LOW, SCHEDULE_HIGH_LOW),
    ("hi→lo matvec  / lo→hi orth",        SCHEDULE_HIGH_LOW, SCHEDULE_LOW_HIGH, SCHEDULE_LOW_HIGH),
    ("lo→hi BOTH    (same schedule)",     SCHEDULE_LOW_HIGH, SCHEDULE_LOW_HIGH, SCHEDULE_LOW_HIGH),
    ("hi→lo mv, lo→hi push / hi→lo pop", SCHEDULE_HIGH_LOW, SCHEDULE_LOW_HIGH, SCHEDULE_HIGH_LOW),
]

styles = [:solid, :dash, :dot, :dashdot, :solid, :dash, :dot, :dashdot]
colors = [:black, :blue, :red, :green, :cyan, :orange, :purple, :brown]

plot_kw = (yscale=:log10, lw=1.5, legendfontsize=7, titlefontsize=9,
           tickfontsize=7, guidefontsize=8, framestyle=:box,
           bottom_margin=5mm, left_margin=3mm)

println("="^80)
println("  Adaptive-precision GMRES: matvec and orth schedules vary independently")
println("  n=$n  restart=$restart  tol=$tol  eta=$eta")
println("  Config                                  residual  iters  ✓  mv-formats  orth-formats")
println("="^80)

results = []
for (lbl, mv_s, pu_s, po_s) in configs
    _, errs, terrs, orth, mv_f, pu_f = relaxed_gmres_adaptive_orth(
        A, b, x0, max_iter, restart, tol, eta;
        mv_schedule=mv_s, orth_push_schedule=pu_s, orth_pop_schedule=po_s)
    res = terrs[end] / nrm_b
    mv_summary  = join(["$(f):$(count(==(f),mv_f))"  for f in unique(mv_f)], " ")
    pu_summary  = join(["$(f):$(count(==(f),pu_f))"  for f in unique(pu_f)], " ")
    @printf("  %-40s  %.2e  %4d  %s  [%s]  [%s]\n",
            lbl, res, length(errs)-1, res<tol ? "✓" : "✗", mv_summary, pu_summary)
    push!(results, (lbl, errs, terrs, orth))
end
println("="^80)

p1 = plot(; title="Projected Krylov Residual", xlabel="iteration",
            ylabel="residual / ‖b‖", plot_kw...)
p2 = plot(; title="True Residual  ‖b-Ax‖/‖b‖", xlabel="iteration",
            ylabel="", plot_kw...)
p3 = plot(; title="Orthogonality  ‖I - Q'Q‖_fro", xlabel="iteration",
            ylabel="", plot_kw...)

for (i, (lbl, errs, terrs, orth)) in enumerate(results)
    plot!(p1, 0:length(errs)-1,  errs  ./ nrm_b, label=lbl, ls=styles[i], color=colors[i])
    plot!(p2, 0:length(terrs)-1, terrs ./ nrm_b, label=lbl, ls=styles[i], color=colors[i])
    plot!(p3, 0:length(orth)-1,  max.(orth, eps(Float64)), label=lbl, ls=styles[i], color=colors[i])
end
hline!(p1, [tol],          ls=:dash, color=:grey, label="tol",      lw=1)
hline!(p2, [tol],          ls=:dash, color=:grey, label="tol",      lw=1)
hline!(p3, [eps(Float32)], ls=:dot,  color=:blue, label="F32 eps",  lw=1)
hline!(p3, [eps(Float16)], ls=:dot,  color=:red,  label="F16 eps",  lw=1)

sz = (1000, 520)
f1 = tempname() * "_adaptive_orth_residuals.png"
f2 = tempname() * "_adaptive_orth_orthogonality.png"

savefig(plot(p1, p2; layout=(1,2), size=sz,
             plot_title="Adaptive-precision GMRES  n=$n restart=$restart tol=$tol"), f1)
savefig(plot(p3; size=(700,520),
             plot_title="Krylov Basis Orthogonality"), f2)

println("Saved: $f1"); println("Saved: $f2")
run(`open $f1 $f2`)
