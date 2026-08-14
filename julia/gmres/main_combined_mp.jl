include("restarted_gmres.jl")
include("restarted_gmres_mp.jl")
include("relaxed_gmres.jl")
include("relaxed_gmres_low.jl")
include("relaxed_gmres_orth_mp.jl")
using LinearAlgebra, SparseArrays, Plots, Printf, Random, Measures

Random.seed!(42)
n  = 60
F  = qr(randn(n, n))
Qm = Matrix(F.Q)
S  = Diagonal(10 .^ LinRange(0, 2, n))   # cond(A) ≈ 100
A  = Qm * Matrix(S) * Qm'
b  = randn(n)
x0 = zeros(n)

max_iter = 400
restart  = 400
tol      = 1e-10
nrm_b    = norm(b)

styles = [:solid, :dash, :dot, :dashdot, :solid, :dash]
colors = [:black, :blue, :red, :green, :cyan, :orange]

plot_kw = (yscale=:log10, lw=1.5, legendfontsize=7, titlefontsize=9,
           tickfontsize=7, guidefontsize=8, framestyle=:box,
           top_margin=2mm, bottom_margin=2mm, left_margin=3mm)

function row_plots(results, row_label)
    p1 = plot(; title="$row_label — projected", ylabel="residual/‖b‖",
                xlabel="", plot_kw...)
    p2 = plot(; title="$row_label — true ‖b-Ax‖/‖b‖",
                ylabel="", xlabel="", plot_kw...)
    for (i, (lbl, errs, terrs)) in enumerate(results)
        plot!(p1, 0:length(errs)-1,  errs  ./ nrm_b, label=lbl,
              ls=styles[i], color=colors[i])
        plot!(p2, 0:length(terrs)-1, terrs ./ nrm_b, label=lbl,
              ls=styles[i], color=colors[i])
    end
    hline!(p1, [tol], ls=:dot, color=:grey, label="tol", lw=1)
    hline!(p2, [tol], ls=:dot, color=:grey, label="tol", lw=1)
    return p1, p2
end

# ── Row 1: MP Householder, F64 matvec ────────────────────────────────────────
mp_configs = [
    ("F64 (base)",  Float64, Float64, Float64),
    ("F32s/F64a",   Float32, Float64, Float64),
    ("F64s/F32a",   Float64, Float32, Float64),
    ("F32s/F32a",   Float32, Float32, Float32),
    ("F16s/F64a",   Float16, Float64, Float64),
    ("F64s/F16a",   Float64, Float16, Float64),
]

println("── Row 1: MP Householder orth (F64 matvec) ─────────────────────────────")
mp_res = []
for (lbl, Ts, Ta, Tr) in mp_configs
    _, errs, terrs = restarted_gmres_mp(A, b, x0, max_iter, restart, tol;
                                         T_store=Ts, T_apply=Ta, T_reflect=Tr)
    res = terrs[end] / nrm_b
    @printf("  %-18s  %.2e  %4d  %s\n", lbl, res, length(errs)-1,
            res < tol ? "✓" : "✗")
    push!(mp_res, (lbl, errs, terrs))
end

# ── Row 2: Relaxed matvec, F64 Householder ───────────────────────────────────
rlx_configs = [
    ("F64 (base)",    :none,     0.0),
    ("hi→lo η=0.01",  :high2low, 0.01),
    ("hi→lo η=0.1",   :high2low, 0.1),
    ("hi→lo η=1.0",   :high2low, 1.0),
    ("lo→hi η=0.1",   :low2high, 0.1),
    ("lo→hi η=1.0",   :low2high, 1.0),
]

println("── Row 2: Relaxed matvec (F64 Householder) ─────────────────────────────")
rlx_res = []
for (lbl, mode, eta) in rlx_configs
    if mode == :none
        _, errs, terrs = restarted_gmres(A, b, x0, max_iter, restart, tol)
    elseif mode == :high2low
        _, errs, _, terrs = relaxed_gmres(A, b, x0, max_iter, restart, tol, eta)
    else
        _, errs, _, terrs = relaxed_gmres_low(A, b, x0, max_iter, restart, tol, eta)
    end
    res = terrs[end] / nrm_b
    @printf("  %-18s  %.2e  %4d  %s\n", lbl, res, length(errs)-1,
            res < tol ? "✓" : "✗")
    push!(rlx_res, (lbl, errs, terrs))
end

# ── Row 3: Combined — relaxed matvec + MP Householder ────────────────────────
comb_configs = [
    ("F64 (base)",          :high2low, 0.0,  Float64, Float64, Float64),
    ("hi→lo + F32s/F64a",   :high2low, 0.1,  Float32, Float64, Float64),
    ("hi→lo + F32s/F32a",   :high2low, 0.1,  Float32, Float32, Float32),
    ("lo→hi + F32s/F64a",   :low2high, 1.0,  Float32, Float64, Float64),
    ("lo→hi + F64s/F32a",   :low2high, 1.0,  Float64, Float32, Float64),
    ("lo→hi + F32s/F32a",   :low2high, 1.0,  Float32, Float32, Float32),
]

println("── Row 3: Combined relaxed matvec + MP Householder ─────────────────────")
comb_res = []
for (lbl, sched, eta, Ts, Ta, Tr) in comb_configs
    if eta == 0.0
        _, errs, terrs = restarted_gmres(A, b, x0, max_iter, restart, tol)
    else
        _, errs, terrs, _ = relaxed_gmres_orth_mp(A, b, x0, max_iter, restart, tol, eta;
                                                    matvec_schedule=sched,
                                                    T_store=Ts, T_apply=Ta, T_reflect=Tr)
    end
    res = terrs[end] / nrm_b
    @printf("  %-26s  %.2e  %4d  %s\n", lbl, res, length(errs)-1,
            res < tol ? "✓" : "✗")
    push!(comb_res, (lbl, errs, terrs))
end

# ── Save each row as a PNG and open all in Preview simultaneously ─────────────
p1, p2 = row_plots(mp_res,   "MP Householder (s=store a=apply)")
p3, p4 = row_plots(rlx_res,  "Relaxed matvec")
p5, p6 = row_plots(comb_res, "Combined")

for p in (p1, p2, p3, p4, p5, p6)
    plot!(p; xlabel="iteration", bottom_margin=5mm)
end

sz = (900, 450)   # comfortable single-row size

row1 = plot(p1, p2; layout=(1,2), size=sz,
            plot_title="Row 1 — MP Householder orth  (F64 matvec)")
row2 = plot(p3, p4; layout=(1,2), size=sz,
            plot_title="Row 2 — Relaxed matvec  (F64 Householder)")
row3 = plot(p5, p6; layout=(1,2), size=sz,
            plot_title="Row 3 — Combined relaxed matvec + MP Householder")

f1 = tempname() * "_row1_mp_householder.png"
f2 = tempname() * "_row2_relaxed_matvec.png"
f3 = tempname() * "_row3_combined.png"

savefig(row1, f1); println("Saved: $f1")
savefig(row2, f2); println("Saved: $f2")
savefig(row3, f3); println("Saved: $f3")

# Open all three simultaneously in macOS Preview (tiles side-by-side).
run(`open $f1 $f2 $f3`)
