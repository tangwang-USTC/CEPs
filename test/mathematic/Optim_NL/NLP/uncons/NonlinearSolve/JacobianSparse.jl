




using NonlinearSolve, LinearAlgebra, SparseArrays, LinearSolve
using BenchmarkTools

const N = 32
const xyd_brusselator = range(0, stop = 1, length = N)

brusselator_f(x, y) = (((x - 0.3)^2 + (y - 0.6)^2) <= 0.1^2) * 5.0
limit(a, N) = a == N + 1 ? 1 : a == 0 ? N : a

function brusselator_2d_loop(du, u, p)
    A, B, alpha, dx = p
    alpha = alpha / dx^2
    @inbounds for I in CartesianIndices((N, N))
        i, j = Tuple(I)
        x, y = xyd_brusselator[I[1]], xyd_brusselator[I[2]]
        ip1, im1, jp1, jm1 = limit(i + 1, N), limit(i - 1, N), limit(j + 1, N),
        limit(j - 1, N)
        du[i, j, 1] = alpha * (u[im1, j, 1] + u[ip1, j, 1] + u[i, jp1, 1] + u[i, jm1, 1] -
                       4u[i, j, 1]) +
                      B +
                      u[i, j, 1]^2 * u[i, j, 2] - (A + 1) * u[i, j, 1] + brusselator_f(x, y)
        du[i, j, 2] = alpha * (u[im1, j, 2] + u[ip1, j, 2] + u[i, jp1, 2] + u[i, jm1, 2] -
                       4u[i, j, 2]) + A * u[i, j, 1] - u[i, j, 1]^2 * u[i, j, 2]
    end
end
p = (3.4, 1.0, 10.0, step(xyd_brusselator))

function init_brusselator_2d(xyd)
    N = length(xyd)
    u = zeros(N, N, 2)
    for I in CartesianIndices((N, N))
        x = xyd[I[1]]
        y = xyd[I[2]]
        u[I, 1] = 22 * (y * (1 - y))^(3 / 2)
        u[I, 2] = 27 * (x * (1 - x))^(3 / 2)
    end
    u
end

u0 = init_brusselator_2d(xyd_brusselator)
prob_brusselator_2d = NonlinearProblem(
    brusselator_2d_loop, u0, p; abstol = 1e-10, reltol = 1e-10
)


############### Jacobian types ∈ [sparse,         Bidiagonal,Tridiagonal,SymTridiagonal,BandedMatrix,BlockBandedMatrix]
############### Jacobian types ∈ [SparseMatrixCSC,Bidiagonal,Tridiagonal,SymTridiagonal,BandedMatrix.jl,BlockBandedMatrix.jl]

@btime solve(prob_brusselator_2d, NewtonRaphson())




using SparseConnectivityTracer

prob_brusselator_2d_autosparse = NonlinearProblem(
    NonlinearFunction(brusselator_2d_loop; sparsity = TracerSparsityDetector()),
    u0, p; abstol = 1e-10, reltol = 1e-10
)

@btime solve(prob_brusselator_2d_autosparse,
    NewtonRaphson(; autodiff = AutoForwardDiff(; chunksize = 12),
        linsolve = KrylovJL_GMRES()));                                               # perfect
@btime solve(prob_brusselator_2d_autosparse,
    NewtonRaphson(; autodiff = AutoForwardDiff(; chunksize = 12)));
@btime solve(prob_brusselator_2d_autosparse,
    NewtonRaphson(; autodiff = AutoForwardDiff(; chunksize = 12),
        linsolve = KLUFactorization()));

1
using SparseConnectivityTracer, ADTypes

f! = (du, u) -> brusselator_2d_loop(du, u, p)
du0 = similar(u0)
jac_sparsity = ADTypes.jacobian_sparsity(f!, du0, u0, TracerSparsityDetector())

ff = NonlinearFunction(brusselator_2d_loop; jac_prototype = jac_sparsity)
prob_brusselator_2d_sparse = NonlinearProblem(ff, u0, p; abstol = 1e-10, reltol = 1e-10)

println("Sparse")
@btime solve(prob_brusselator_2d, NewtonRaphson());
@btime solve(prob_brusselator_2d_sparse, NewtonRaphson());
@btime solve(prob_brusselator_2d_sparse, NewtonRaphson(linsolve = KLUFactorization()));

println("GMRES")
# Switching to a Krylov linear solver will automatically change the nonlinear problem solver into Jacobian-free mode, dramatically reducing the memory required. 
@btime solve(prob_brusselator_2d, NewtonRaphson(linsolve = KrylovJL_GMRES()));               


# ######################################### Adding a preconditioner         
using IncompleteLU

incompletelu(W, p = nothing) = ilu(W, τ = 50.0), LinearAlgebra.I

println("GMRES_ILU")
@btime solve(prob_brusselator_2d_sparse,
    NewtonRaphson(linsolve = KrylovJL_GMRES(precs = incompletelu), concrete_jac = true)
);
nothing # hide

using AlgebraicMultigrid

function algebraicmultigrid(W, p = nothing)
    return aspreconditioner(ruge_stuben(convert(AbstractMatrix, W))), LinearAlgebra.I
end

println("GMRES_AMG")
@btime solve(prob_brusselator_2d_sparse,
    NewtonRaphson(
        linsolve = KrylovJL_GMRES(; precs = algebraicmultigrid), concrete_jac = true
    )
);

function algebraicmultigrid2(W, p = nothing)
    A = convert(AbstractMatrix, W)
    Pl = AlgebraicMultigrid.aspreconditioner(AlgebraicMultigrid.ruge_stuben(
        A, presmoother = AlgebraicMultigrid.Jacobi(rand(size(A, 1))),
        postsmoother = AlgebraicMultigrid.Jacobi(rand(size(A, 1)))
    ))
    return Pl, LinearAlgebra.I
end

println("GMRES_AMG_Jacobi")
@btime solve(
    prob_brusselator_2d_sparse,
    NewtonRaphson(
        linsolve = KrylovJL_GMRES(precs = algebraicmultigrid2), concrete_jac = true
    )
);


using DifferentiationInterface, SparseConnectivityTracer

prob_brusselator_2d_exact_tracer = NonlinearProblem(
    NonlinearFunction(brusselator_2d_loop; sparsity = TracerSparsityDetector()),
    u0, p; abstol = 1e-10, reltol = 1e-10)
prob_brusselator_2d_approx_di = NonlinearProblem(
    NonlinearFunction(brusselator_2d_loop;
        sparsity = DenseSparsityDetector(AutoForwardDiff(); atol = 1e-4)),
    u0, p; abstol = 1e-10, reltol = 1e-10)

println("Sparsity detection")
@btime solve(prob_brusselator_2d_exact_tracer, NewtonRaphson());
@btime solve(prob_brusselator_2d_approx_di, NewtonRaphson());
1