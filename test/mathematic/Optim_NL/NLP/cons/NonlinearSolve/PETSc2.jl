using NonlinearSolve, LinearAlgebra, SparseConnectivityTracer, BenchmarkTools

u0 = fill(0.5, 128)

function form_residual!(resid, x, _)
    n = length(x)
    xp = LinRange(0.0, 1.0, n)
    F = 6xp .+ (xp .+ 1e-12) .^ 6

    dx = 1 / (n - 1)
    resid[1] = x[1]
    for i in 2:(n - 1)
        resid[i] = (x[i - 1] - 2x[i] + x[i + 1]) / dx^2 + x[i] * x[i] - F[i]
    end
    resid[n] = x[n] - 1

    return
end

nlfunc_dense = NonlinearFunction(form_residual!)
nlfunc_sparse = NonlinearFunction(form_residual!; sparsity = TracerSparsityDetector())

nlprob_dense = NonlinearProblem(nlfunc_dense, u0)
nlprob_sparse = NonlinearProblem(nlfunc_sparse, u0)

sol_dense_nr = solve(nlprob_dense, NewtonRaphson(); abstol = 1e-8)

sol_sparse_nr = solve(nlprob_sparse, NewtonRaphson(); abstol = 1e-8)

if 1 == 2

    @benchmark solve($(nlprob_dense), $(NewtonRaphson()); abstol = 1e-8)
    
    @benchmark solve($(nlprob_sparse), $(NewtonRaphson()); abstol = 1e-8)
end

sol_sparse_nr .- sol_dense_nr

