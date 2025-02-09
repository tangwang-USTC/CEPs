

using NonlinearSolve, BenchmarkTools, SteadyStateDiffEq
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

p = 2.0
u0 = [1.0, 1.0]
u0_SA = SA[1.0, 1.0]

is_SA = true
is_SA = false

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
            prob = NonlinearLeastSquaresProblem(NonlinearFunction(fnls!, resid_prototype = zeros(2)), u0, p)
        end
        sol = solve(prob, TrustRegion())
        # sol = solve(prob, TrustRegion();reltol=1e-12,abstol=1e-12)
    end
end
# @benchmark sol = solve(prob, NewtonRaphson())
@benchmark sols = solve(prob, SimpleNewtonRaphson())

# sol.u
# sol.resid
# sol.retcode          # SciMLBase.successful_retcode(sol)
# sol.alg
# sol.left
# sol.right
# sol.stats


