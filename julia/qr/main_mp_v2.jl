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
fn(T) = T == Float64 ? "F64" : T == Float32 ? "F32" : T == Float16 ? "F16" : "BF16"

# Focus: T_store × T_push (or T_ip for GS), all else fixed at Float64.
# Rows 1-4: vary push precision, T_store=F64.
# Rows 5-8: vary storage precision, T_push=F64.
# Row 6 overlaps (F32/F32) to connect the two sweeps.
push_types  = [Float64, Float32, Float16, BFloat16]
store_types = [Float32, Float16, BFloat16]   # F64 store already covered in rows 1-4

# GS configs: (label, T_store, T_ip)  — T_axpy=T_norm=Float64
cfgs_gs = vcat(
    [("s=F64  ip=$(fn(Tp))", Float64,  Tp) for Tp in push_types],
    [("s=$(fn(Ts)) ip=F64",  Ts, Float64)  for Ts in store_types],
)

# HH configs: (label, T_store, T_push)  — T_pop=T_construct=Float64
cfgs_hh = vcat(
    [("s=F64  push=$(fn(Tp))", Float64,  Tp) for Tp in push_types],
    [("s=$(fn(Ts)) push=F64",  Ts, Float64)  for Ts in store_types],
)

orth_methods = [
    ("CGS",  qr__orth_cgs_mp,      cfgs_gs),
    ("CGS2", qr__orth_cgs2_mp,     cfgs_gs),
    ("MGS",  qr__orth_mgs_lvl1_mp, cfgs_gs),
    ("HH",   qr__orth_hh_lvl1_mp,  cfgs_hh),
]

# Colours: rows 1-4 (vary push) in blue tones, rows 5-7 (vary store) in red tones
styles = [:solid, :solid, :solid, :solid, :dash, :dash, :dash]
colors = [:black, :blue, :cyan, :purple, :red, :orange, :green]

plot_kw = (xscale=:log10, yscale=:log10, lw=1.5, legendfontsize=7, titlefontsize=9,
           tickfontsize=7, guidefontsize=8, framestyle=:box,
           bottom_margin=5mm, left_margin=3mm)

clamp_plot(v) = max.(min.(v, 1.0), eps(Float64))

println("="^70)
println("  Mixed-precision QR v2: storage vs compute precision  m=$m  n=$n")
println("  Fixed: T_pop=T_construct=T_axpy=T_norm=Float64")
println("="^70)

x_axis = 10.0 .^ collect(range_log10KA)
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

        for (c, (_, Ts, Tpu)) in enumerate(precision_configs)
            # GS: T_store=Ts, T_ip=Tpu, T_axpy=T_norm=Float64
            # HH: T_store=Ts, T_push=Tpu, T_pop=T_construct=Float64
            Q, R = if qr_fn === qr__orth_hh_lvl1_mp
                qr_fn(A; T_store=Ts, T_push=Tpu, T_pop=Float64, T_construct=Float64)
            else
                qr_fn(A; T_store=Ts, T_ip=Tpu, T_axpy=Float64, T_norm=Float64)
            end
            Q64 = Float64.(Q); R64 = Float64.(R); A64 = Float64.(A)
            orth_curves[c][k]   = norm(reshape(Matrix(I,n,n) .- Q64'*Q64, :))
            repres_curves[c][k] = norm(A64 .- Q64*R64) / norm(A64)
        end
    end
    @printf("  %s done\n", mth_lbl)

    p_orth   = plot(; title="$mth_lbl — Orthogonality", xlabel="κ(A)",
                     ylabel="‖I - Q'Q‖_fro", plot_kw...)
    p_repres = plot(; title="$mth_lbl — Representation", xlabel="κ(A)",
                     ylabel="‖A-QR‖/‖A‖", plot_kw...)

    for (c, (cfg_lbl, _, _)) in enumerate(precision_configs)
        plot!(p_orth,   x_axis, clamp_plot(orth_curves[c]),   label=cfg_lbl,
              ls=styles[c], color=colors[c])
        plot!(p_repres, x_axis, clamp_plot(repres_curves[c]), label=cfg_lbl,
              ls=styles[c], color=colors[c])
    end

    eps64 = eps(Float64); eps32 = eps(Float32)
    epsbf = Float64(eps(BFloat16))
    plot!(p_orth, x_axis, clamp_plot(x_axis .^ 2 .* eps64),          ls=:dot, color=:grey,    label="κ²·F64eps", lw=1)
    plot!(p_orth, x_axis, clamp_plot(x_axis .^ 2 .* eps32),          ls=:dot, color=:lightblue, label="κ²·F32eps", lw=1)
    plot!(p_orth, x_axis, clamp_plot(x_axis .^ 2 .* Float64(eps(Float16))), ls=:dot, color=:salmon, label="κ²·F16eps", lw=1)
    plot!(p_orth, x_axis, clamp_plot(x_axis .^ 2 .* epsbf),          ls=:dot, color=:magenta,  label="κ²·BF16eps", lw=1)

    sz = (1200, 520)
    row = plot(p_orth, p_repres; layout=(1,2), size=sz,
               plot_title="QR storage vs compute — $mth_lbl  m=$m n=$n")
    f = tempname() * "_qr_mpv2_$(lowercase(mth_lbl)).png"
    savefig(row, f); println("  Saved: $f")
    push!(files, f)
end

run(Cmd(vcat(["open"], files)))
