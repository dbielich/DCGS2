include("qr__orth_cgs_mp.jl")
include("qr__orth_cgs2_mp.jl")
include("qr__orth_mgs_lvl1_mp.jl")
include("qr__orth_hh_lvl1_mp.jl")
include("qr__orth_stabilitymetric.jl")
using LinearAlgebra, Plots, Printf, Random, Measures, BFloat16s

Random.seed!(0)
m = 500; n = 100
range_log10KA = 0:1:16
npts = length(range_log10KA)
fn(T) = T == Float64 ? "F64" : T == Float32 ? "F32" : "F16"

# Gram-Schmidt methods: vary T_ip / T_axpy / T_norm
cfgs_gs = [
    ("F64 all",       (T_ip=Float64,   T_axpy=Float64,   T_norm=Float64)),
    ("F32 ip",        (T_ip=Float32,   T_axpy=Float64,   T_norm=Float64)),
    ("F32 axpy",      (T_ip=Float64,   T_axpy=Float32,   T_norm=Float64)),
    ("F32 norm",      (T_ip=Float64,   T_axpy=Float64,   T_norm=Float32)),
    ("F32 ip+axpy",   (T_ip=Float32,   T_axpy=Float32,   T_norm=Float64)),
    ("F32 all",       (T_ip=Float32,   T_axpy=Float32,   T_norm=Float32)),
    ("F16 ip",        (T_ip=Float16,   T_axpy=Float64,   T_norm=Float64)),
    ("F16 all",       (T_ip=Float16,   T_axpy=Float16,   T_norm=Float16)),
    ("BF16 ip",       (T_ip=BFloat16,  T_axpy=Float64,   T_norm=Float64)),
    ("BF16 all",      (T_ip=BFloat16,  T_axpy=BFloat16,  T_norm=BFloat16)),
]

# Householder: vary T_push / T_pop / T_construct
cfgs_hh = [
    ("F64 all",       (T_push=Float64,  T_pop=Float64,  T_construct=Float64)),
    ("F32 push",      (T_push=Float32,  T_pop=Float64,  T_construct=Float64)),
    ("F32 pop",       (T_push=Float64,  T_pop=Float32,  T_construct=Float64)),
    ("F32 construct", (T_push=Float64,  T_pop=Float64,  T_construct=Float32)),
    ("F32 push+pop",  (T_push=Float32,  T_pop=Float32,  T_construct=Float64)),
    ("F32 all",       (T_push=Float32,  T_pop=Float32,  T_construct=Float32)),
    ("F16 push",      (T_push=Float16,  T_pop=Float64,  T_construct=Float64)),
    ("F16 all",       (T_push=Float16,  T_pop=Float16,  T_construct=Float16)),
    ("BF16 push",     (T_push=BFloat16, T_pop=Float64,  T_construct=Float64)),
    ("BF16 all",      (T_push=BFloat16, T_pop=BFloat16, T_construct=BFloat16)),
]

# Orth methods: (label, qr_fn, precision_configs)
orth_methods = [
    ("CGS",  qr__orth_cgs_mp,      cfgs_gs),
    ("CGS2", qr__orth_cgs2_mp,     cfgs_gs),
    ("MGS",  qr__orth_mgs_lvl1_mp, cfgs_gs),
    ("HH",   qr__orth_hh_lvl1_mp,  cfgs_hh),
]

styles = [:solid, :dash, :dot, :dashdot, :solid, :dash, :dot, :dashdot, :solid, :dash]
colors = [:black, :blue, :red, :green, :cyan, :orange, :purple, :brown, :pink, :teal]

plot_kw = (xscale=:log10, yscale=:log10, lw=1.5, legendfontsize=7, titlefontsize=9,
           tickfontsize=7, guidefontsize=8, framestyle=:box,
           bottom_margin=5mm, left_margin=3mm)

println("="^70)
println("  Mixed-precision QR: condition number sweep  m=$m  n=$n")
println("  Methods: $(join([m[1] for m in orth_methods], ", "))")
println("="^70)

x_axis = 10.0 .^ collect(range_log10KA)   # Float64 prevents Int64 overflow in x^2

# One PNG per orth method, each showing orthogonality and repr error
files = String[]
for (mth_lbl, qr_fn, precision_configs) in orth_methods
    orth_curves   = [zeros(npts) for _ in precision_configs]
    repres_curves = [zeros(npts) for _ in precision_configs]
    cond_vals     = zeros(npts)

    for (k, log10KA) in enumerate(range_log10KA)
        U, _ = qr(randn(m, n)); V, _ = qr(randn(n, n))
        S = Diagonal(10 .^ LinRange(0, log10KA, n))
        A = Matrix(U) * Matrix(S) * Matrix(V)'
        cond_vals[k] = cond(A)

        for (c, (_, kwargs)) in enumerate(precision_configs)
            Q, R = qr_fn(A; kwargs...)
            Q64 = Float64.(Q); R64 = Float64.(R); A64 = Float64.(A)
            orth_curves[c][k]   = norm(reshape(Matrix(I,n,n) .- Q64'*Q64, :))
            repres_curves[c][k] = norm(A64 .- Q64*R64) / norm(A64)
        end
    end

    @printf("  %s done\n", mth_lbl)

    p_orth  = plot(; title="$mth_lbl — Orthogonality", xlabel="κ(A)",
                    ylabel="‖I - Q'Q‖_fro", plot_kw...)
    p_repres = plot(; title="$mth_lbl — Representation", xlabel="κ(A)",
                     ylabel="‖A-QR‖/‖A‖", plot_kw...)

    # clamp helper: floor at eps(Float64), ceiling at 1 (orthogonality/repr can't exceed O(1))
    clamp_plot(v) = max.(min.(v, 1.0), eps(Float64))

    for (c, (cfg_lbl, _)) in enumerate(precision_configs)
        plot!(p_orth,   x_axis, clamp_plot(orth_curves[c]),   label=cfg_lbl,
              ls=styles[c], color=colors[c])
        plot!(p_repres, x_axis, clamp_plot(repres_curves[c]), label=cfg_lbl,
              ls=styles[c], color=colors[c])
    end

    # Reference lines: κ²·eps and κ·eps clamped at 1 (bounds are meaningless above O(1))
    eps64 = eps(Float64); eps32 = eps(Float32); eps16 = eps(Float16)
    epsbf = Float64(eps(BFloat16))   # BF16 eps ≈ 7.8e-3, saturates at κ ≈ 11
    plot!(p_orth, x_axis, clamp_plot(x_axis .^ 2 .* eps64),  ls=:dot, color=:grey,
          label="κ²·F64eps", lw=1)
    plot!(p_orth, x_axis, clamp_plot(x_axis .* eps64),       ls=:dot, color=:black,
          label="κ·F64eps",  lw=1)
    plot!(p_orth, x_axis, clamp_plot(x_axis .^ 2 .* eps32),  ls=:dot, color=:lightblue,
          label="κ²·F32eps", lw=1)
    plot!(p_orth, x_axis, clamp_plot(x_axis .^ 2 .* Float64(eps16)), ls=:dot, color=:salmon,
          label="κ²·F16eps", lw=1)
    plot!(p_orth, x_axis, clamp_plot(x_axis .^ 2 .* epsbf),  ls=:dot, color=:magenta,
          label="κ²·BF16eps", lw=1)

    sz = (1100, 500)
    row = plot(p_orth, p_repres; layout=(1,2), size=sz,
               plot_title="QR mixed precision — $mth_lbl  m=$m n=$n")
    f = tempname() * "_qr_mp_$(lowercase(mth_lbl)).png"
    savefig(row, f); println("  Saved: $f")
    push!(files, f)
end

run(Cmd(vcat(["open"], files)))
