# Matrix Market reader for Julia.
# The MatrixMarket.jl package handles .mtx files natively and is the
# preferred approach. Install once with:
#   using Pkg; Pkg.add("MatrixMarket")
#
# Usage (mirrors mmread.m):
#   using MatrixMarket
#   A = mmread("../matrix_market/thermal1.mtx")
#
# If you need a self-contained fallback without the package, the function
# below reads the common real/integer coordinate and array formats.

using SparseArrays, LinearAlgebra

function mmread(filename::AbstractString)
    open(filename, "r") do io
        header = readline(io)
        if !startswith(header, "%%MatrixMarket")
            error("Not a Matrix Market file: $filename")
        end
        tokens = lowercase.(split(header))
        is_sparse  = tokens[4] == "coordinate"
        is_complex = length(tokens) >= 5 && tokens[5] == "complex"
        is_pattern = length(tokens) >= 5 && tokens[5] == "pattern"
        is_symm    = length(tokens) >= 6 && tokens[6] in ("symmetric", "skew-symmetric", "hermitian")
        is_skew    = length(tokens) >= 6 && tokens[6] == "skew-symmetric"

        # Skip comment lines
        line = readline(io)
        while startswith(line, "%")
            line = readline(io)
        end

        dims = parse.(Int, split(line))
        if is_sparse
            m, n, nnz = dims
            I_idx = Int[]
            J_idx = Int[]
            vals  = Float64[]
            for _ in 1:nnz
                parts = split(readline(io))
                push!(I_idx, parse(Int, parts[1]))
                push!(J_idx, parse(Int, parts[2]))
                v = is_pattern ? 1.0 : parse(Float64, parts[3])
                push!(vals, v)
            end
            A = sparse(I_idx, J_idx, vals, m, n)
            if is_symm
                A = A + A' - sparse(Diagonal(diag(A)))
            elseif is_skew
                A = A - A'
            end
            return A
        else
            m, n = dims
            A = zeros(m, n)
            for j in 1:n, i in 1:m
                A[i, j] = parse(Float64, split(readline(io))[1])
            end
            return A
        end
    end
end
