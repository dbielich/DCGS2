include("arnoldi__orth_hh_lvl1.jl")
include("arnoldi__orth_hh_lvl1_mp.jl")
include("arnoldi__orth_stabilitymetric.jl")
using LinearAlgebra, Plots, Printf

# Experiment matrix: diagonal with one tiny eigenvalue — stresses orthogonality.
m = 100; k = 80
A = Diagonal(LinRange(1, m, m)); A = Matrix(A); A[1,1] = 1e-8
b = randn(m)
A = sparse(A)

configs = [
    # (label,                         T_store,  T_apply,  T_reflect)
    ("F64 all (baseline)",            Float64,  Float64,  Float64),
    ("F32 store / F64 apply (Tisseur)",Float32, Float64,  Float64),   # cheap storage, accurate application
    ("F64 store / F32 apply",         Float64,  Float32,  Float64),   # accurate storage, cheap application
    ("F32 store / F32 apply",         Float32,  Float32,  Float32),   # full low-precision
    ("F16 store / F64 apply",         Float16,  Float64,  Float64),   # very cheap storage, accurate application
    ("F64 store / F16 apply",         Float64,  Float16,  Float64),   # accurate storage, very cheap application
]

labels    = String[]
orth_vals = Vector{Float64}[]
repr_vals = Vector{Float64}[]

println("="^70)
println("  Config                              final_orth    final_repr")
println("="^70)

for (lbl, Ts, Ta, Tr) in configs
    Q, H, beta = arnoldi__orth_hh_lvl1_mp(A, b, k;
                                           T_store=Ts, T_apply=Ta, T_reflect=Tr)
    repres, orth, _, _ = arnoldi__orth_stabilitymetric(A, b, Q, H, beta)
    push!(labels, lbl)
    push!(orth_vals, orth)
    push!(repr_vals, repres)
    @printf("  %-36s  %.2e    %.2e\n", lbl, orth[end], repres[end])
end
println("="^70)

x = 1:k

p_orth = plot(title="Orthogonality  || I - Q'Q ||_fro", xlabel="Iteration",
              ylabel="", yscale=:log10, legend=:topleft)
p_repr = plot(title="Representation error / ||A||", xlabel="Iteration",
              ylabel="", yscale=:log10, legend=:topleft)

styles = [:solid, :dash, :dot, :dashdot, :solid, :dash]
colors = [:black, :blue, :red, :green, :cyan, :orange]

for i in eachindex(labels)
    plot!(p_orth, x, orth_vals[i], label=labels[i], lw=2,
          ls=styles[i], color=colors[i])
    plot!(p_repr, x, repr_vals[i], label=labels[i], lw=2,
          ls=styles[i], color=colors[i])
end

display(plot(p_orth, p_repr, layout=(1,2), size=(1400, 550),
             plot_title="Mixed-precision Householder Arnoldi\n(tau always Float64)"))
