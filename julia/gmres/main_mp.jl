include("restarted_gmres.jl")
include("restarted_gmres_mp.jl")
using LinearAlgebra, SparseArrays, Plots, Printf, Random

Random.seed!(42)
n  = 60
A  = diagm(0  =>  ones(n),
       1  =>  ones(n-1),
       2  =>  ones(n-2),
       3  =>  ones(n-3),
       -1 => -ones(n-1))
b  = randn(n); x0 = zeros(n)

max_iter = 400; restart = 400; tol = 1e-10
nrm_b    = norm(b)

fn(T) = T == Float64 ? "F64" : T == Float32 ? "F32" : "F16"

# Configs: (T_store, T_work, T_push, T_pop, T_construct)
# store/work: F32 or F64 only; push/pop: all three; construct: matches T_work
configs = [
    (Float64, Float64, Float64, Float64, Float64),   # baseline
    (Float64, Float64, Float32, Float32, Float64),   # F32 push/pop, F64 store/work
    (Float64, Float64, Float16, Float16, Float64),   # F16 push/pop, F64 store/work
    (Float32, Float32, Float64, Float64, Float32),   # F32 store/work, F64 push/pop
    (Float32, Float32, Float32, Float32, Float32),   # all F32
    (Float32, Float32, Float16, Float16, Float32),   # F32 store/work, F16 push/pop
    (Float64, Float64, Float32, Float16, Float64),   # asymmetric: F32 push, F16 pop
    (Float64, Float64, Float16, Float32, Float64),   # asymmetric: F16 push, F32 pop
    (Float64, Float64, Float64, Float32, Float64),   # asymmetric: F64 push, F32 pop
    (Float64, Float64, Float32, Float64, Float64),   # asymmetric: F32 push, F64 pop
    (Float32, Float64, Float64, Float32, Float64),   # F32 store, F64 work/push, F32 pop
]

lbl(Ts, Tw, Tpu, Tpo, Tc) =
    "s=$(fn(Ts)) w=$(fn(Tw)) push=$(fn(Tpu)) pop=$(fn(Tpo)) c=$(fn(Tc))"

results = []

println("="^80)
println("  Config                                     residual   iters  converged?")
println("="^80)
for (Ts, Tw, Tpu, Tpo, Tc) in configs
    x_mp, errs, terrs, orth = restarted_gmres_mp(A, b, x0, max_iter, restart, tol;
                                            T_store=Ts, T_work=Tw,
                                            T_push=Tpu, T_pop=Tpo, T_construct=Tc)
    res  = norm(b .- A * x_mp) / nrm_b
    conv = res < tol ? "yes" : "NO"
    lab  = lbl(Ts, Tw, Tpu, Tpo, Tc)
    @printf("  %-42s  %.2e  %4d   %s\n", lab, res, length(errs)-1, conv)
    push!(results, (lab, errs, terrs, orth))
end
println("="^80)

styles = [:solid, :dash, :dot, :dashdot, :solid, :dash, :dot, :dashdot, :solid, :dash, :solid]
colors = [:black, :blue, :red, :green, :cyan, :orange, :purple, :brown, :pink, :teal, :magenta]

p1 = plot(title="Projected Krylov Residual", xlabel="Iteration",
          ylabel="Residual / ||b||", yscale=:log10, legend=:outerright)
p2 = plot(title="True Residual  ||b - Ax|| / ||b||", xlabel="Iteration",
          ylabel="", yscale=:log10, legend=:outerright)
p3 = plot(title="Krylov Basis Orthogonality  ||I - Q'Q||_fro", xlabel="Iteration",
          ylabel="", yscale=:log10, legend=:outerright)

for (i, (lab, errs, terrs, orth)) in enumerate(results)
    plot!(p1, 0:length(errs)-1,  errs  ./ nrm_b, label=lab, lw=2,
          ls=styles[i], color=colors[i])
    plot!(p2, 0:length(terrs)-1, terrs ./ nrm_b, label=lab, lw=2,
          ls=styles[i], color=colors[i])
    plot!(p3, 0:length(orth)-1,  max.(orth, eps(Float64)), label=lab, lw=2,
          ls=styles[i], color=colors[i])
end
hline!(p1, [tol],          ls=:dash, color=:grey, label="tol",      lw=1)
hline!(p2, [tol],          ls=:dash, color=:grey, label="tol",      lw=1)
hline!(p3, [eps(Float32)], ls=:dot,  color=:blue, label="F32 eps",  lw=1)
hline!(p3, [eps(Float16)], ls=:dot,  color=:red,  label="F16 eps",  lw=1)

sz = (1100, 520)
row_res  = plot(p1, p2; layout=(1,2), size=sz,
                plot_title="MP Householder GMRES  n=$n, restart=$restart, tol=$tol")
row_orth = plot(p3; size=(700, 520),
                plot_title="Krylov Basis Orthogonality")

f1 = tempname() * "_mp_residuals.png"
f2 = tempname() * "_mp_orthogonality.png"
savefig(row_res,  f1); println("Saved: $f1")
savefig(row_orth, f2); println("Saved: $f2")
run(`open $f1 $f2`)

display(plot(p1, p2, layout=(1, 2), size=(1600, 600),
             plot_title="MP Householder GMRES  n=$n, restart=$restart, tol=$tol"))

