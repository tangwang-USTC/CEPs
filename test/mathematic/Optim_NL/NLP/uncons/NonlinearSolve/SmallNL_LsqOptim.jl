

using NonlinearSolve, BenchmarkTools, SteadyStateDiffEq, LeastSquaresOptim
using Optimization
# using Krylov
using LinearSolve
using StaticArrays    # length(vector) ≤ 20 or 100, now static arrays can only be used for sufficiently small arrays



f(u, p) = u .* u .- p
# f(u, p) = [u[1] * u[1] - p, u[2] * u[2] - p]
ft(u, p, t) = [u[1] * u[1] - p, u[2] * u[2] - p]       # SSP

f_SA(u, p) = SA[u[1] * u[1] - p, u[2] * u[2] - p]      
ft_SA(u, p, t) = SA[u[1] * u[1] - p, u[2] * u[2] - p]         

function f!(du, u, p)
    du[1] = u[1] * u[1] - p
    du[2] = u[2] * u[2] - p
    return nothing
end

function fnls!(du, u, p)
    du[1] = u[1] * u[1] - p
    du[2] = u[2] * u[2] - p
    return nothing
end

function Jnls!(du, u, p)
    du[1,1] = 2u[1]
    du[1,2] = 0.0
    du[2,1] = 0.0
    du[2,2] = 2u[2]
    return nothing
end

function jvpnls!(du, u, p)
    du[1] = u[1] * u[1] - p
    du[2] = u[2] * u[2] - p
    return nothing
end

function vjpnls!(du, u, p)
    du[1] = u[1] * u[1] - p
    du[2] = u[2] * u[2] - p
    return nothing
end


# nlfun = NonlinearFunction(fnls!; jac=Jnls!,resid_prototype = zeros(2))

# rrrrr

p = 2.0
u0 = [1.0, 1.0]
u0_SA = SA[1.0, 1.0]

is_SA = true
is_SA = false

ubs = [2.0, 2.0]
lbs = [1.0, 1.0]

solve_type = :NLS         # [:NLP, :NLS, :SSP]
# solve_type = :NLP
# solve_type = :SSP
if is_SA 
    # prob = NonlinearProblem(f_SA, u0_SA, p)
    # prob = NonlinearProblem(f_SA, u0, p)
    if solve_type == :SSP
        prob = SteadyStateProblem(ft_SA, u0_SA, p)
        sol = solve(prob, SSRootfind();reltol=1e-12,abstol=1e-12)
    else
        if solve_type == :NLP
            prob = NonlinearProblem(f, u0_SA, p)
        elseif solve_type == :NLS
            prob = NonlinearLeastSquaresProblem(NonlinearFunction(f_SA, resid_prototype = zeros(2)), u0_SA, p)
        end
        sol = solve(prob, TrustRegion())
        # sol = solve(prob, TrustRegion();reltol=1e-12,abstol=1e-12)
    end
else
    if solve_type == :SSP
        prob = SteadyStateProblem(ft, u0, p)
        sol = solve(prob, SSRootfind();reltol=1e-12,abstol=1e-12)
    else
        if solve_type == :NLP
            prob = NonlinearProblem(f, u0, p)
        elseif solve_type == :NLS
            prob = NonlinearLeastSquaresProblem(NonlinearFunction(fnls!; resid_prototype = zeros(2)), u0, p)
            # prob = NonlinearLeastSquaresProblem(fnls!, u0, p)
        end
        sol = NonlinearSolve.solve(prob, TrustRegion())
        sol = NonlinearSolve.solve(prob, NonlinearSolveFirstOrder.TrustRegion())
        # sol = solve(prob, TrustRegion();reltol=1e-12,abstol=1e-12)
    end
end
# @benchmark sol = solve(prob, NewtonRaphson()) 
@benchmark sols = solve(prob, SimpleNewtonRaphson())

solsLsqO = solve(prob, LeastSquaresOptimJL(:lm;linsolve=:qr,autodiff=:central))

a = NonlinearSolveFirstOrder.NewtonRaphson
solsNLS = solve(prob, a(;linsolve = QRFactorization(), linesearch = missing,autodiff = nothing))
solsNLS = solve(prob, a(;linsolve = KrylovJL_CG(), linesearch = missing,autodiff = nothing))

# sol.u
# sol.resid
# sol.retcode          # SciMLBase.successful_retcode(sol)
# sol.alg
# sol.left
# sol.right
# sol.stats


# probf = OptimizationFunction(f, Optimization.AutoForwardDiff())

# solf = Optimization.solve(prob, Ipopt.Optimizer())

@benchmark sols = solve(prob, SimpleNewtonRaphson())
