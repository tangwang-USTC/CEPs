

using NonlinearSolve

f(u, p) = u .* u .- 2.0
u0 = 1.5
probB = NonlinearProblem(f, u0)

nlcache = init(probB, NewtonRaphson())

# step!(
#     cache::AbstractNonlinearSolveCache;
#     recompute_jacobian::Union{Nothing, Bool} = nothing
# )

for i in 1:10
    step!(nlcache)
end




