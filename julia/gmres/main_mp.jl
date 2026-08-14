include("restarted_gmres.jl")
include("restarted_gmres_mp.jl")
using LinearAlgebra, SparseArrays, Plots, Printf, Random

Random.seed!(42)
n  = 60
F  = qr(randn(n, n))
Q  = Matrix(F.Q)
S  = Diagonal(10 .^ LinRange(0, 2, n))   # cond(A) ≈ 100
A  = Q * Matrix(S) * Q'
b  = randn(n)
x0 = zeros(n)

max_iter = 400
restart  = 20
tol      = 1e-10

configs = [
    # (label,                             T_store,  T_apply,  T_reflect)
    ("F64 all (baseline)",               Float64,  Float64,  Float64),
    ("F32 store / F64 apply (Tisseur)",  Float32,  Float64,  Float64),
    ("F64 store / F32 apply",            Float64,  Float32,  Float64),
    ("F32 store / F32 apply",            Float32,  Float32,  Float32),
    ("F16 store / F64 apply",            Float16,  Float64,  Float64),
    ("F64 store / F16 apply",            Float64,  Float16,  Float64),
]

fmtstr(T) = T == Float64 ? "F64" : T == Float32 ? "F32" : "F16"

results = []

println("="^72)
println("  Config                              residual      iters  converged?")
println("="^72)
for (lbl, Ts, Ta, Tr) in configs
    x_mp, errs, terrs = restarted_gmres_mp(A, b, x0, max_iter, restart, tol;
                                            T_store=Ts, T_apply=Ta, T_reflect=Tr)
    res = norm(b .- A * x_mp) / norm(b)
    conv = res < tol ? "yes" : "NO"
    @printf("  %-36s  %.2e    %4d   %s\n", lbl, res, length(errs)-1, conv)
    push!(results, (lbl, errs, terrs))
end
println("="^72)

styles = [:solid, :dash, :dot, :dashdot, :solid, :dash]
colors = [:black, :blue, :red, :green, :cyan, :orange]

p1 = plot(title="Projected Krylov Residual", xlabel="Iteration",
          ylabel="Residual / ||b||", yscale=:log10, legend=:topright)
p2 = plot(title="True Residual  ||b - Ax|| / ||b||", xlabel="Iteration",
          ylabel="", yscale=:log10, legend=:topright)

for (i, (lbl, errs, terrs)) in enumerate(results)
    nrm = norm(b)
    plot!(p1, 0:length(errs)-1,  errs  ./ nrm, label=lbl, lw=2,
          ls=styles[i], color=colors[i])
    plot!(p2, 0:length(terrs)-1, terrs ./ nrm, label=lbl, lw=2,
          ls=styles[i], color=colors[i])
end

hline!(p1, [tol], ls=:dash, color=:grey, label="tol", lw=1)
hline!(p2, [tol], ls=:dash, color=:grey, label="tol", lw=1)

display(plot(p1, p2, layout=(1,2), size=(1400, 550),
             plot_title="Mixed-precision Householder GMRES  (tau always Float64)\ncond(A)≈100, n=$n, restart=$restart"))
