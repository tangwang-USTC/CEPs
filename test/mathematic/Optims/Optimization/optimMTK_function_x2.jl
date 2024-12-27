using ModelingToolkit, SparseArrays, Test, Optimization, OptimizationOptimJL,
      OptimizationMOI, Ipopt, AmplNLWriter, Ipopt_jll
using ModelingToolkit: get_metadata

# @variables x[1:3] = [0.0, 0.0, 0.0]
@variables x[1:3]
@variables x1, x2, x3
x = [x1, x2, x3]

@parameters a b

cons = [1.0 ~ x[1]^2 + x[2]^2
        x[3] ~ x[2] - x[1]^2
        x[3]^2 + x[2]^2 ≲ 1.0]

loss = (a - x[1])^2 + b * x[3]^2

loss = (a - x[1])^2
loss += b * x[3]^2
# @show typeof(loss), loss

function lossfun(x::AbstractVector{Num},a::Num,b::Num)
    
    loss = (a - x[1])^2
    @show typeof(loss), loss
    loss += b * x[3]^2
    @show typeof(loss), loss
    return loss
end

loss = lossfun(x,a,b)
@named sys = OptimizationSystem(loss, x, [a, b]; constraints = cons)


function lossfun!(loss::AbstractVector{Num},x::AbstractVector{Num},a::Num,b::Num)
    
    loss[1] = a - x[1]
    loss[2] = b * x[3]
    
    return loss
end

loss = zeros(Num,2)
lossfun!(loss,x,a,b)
@named sys = OptimizationSystem(sum(abs2,loss), x, [a, b]; constraints = cons)

sys = structural_simplify(sys)

lbs = [-1.0, -1.0]
ubs = [1.0, 0.01]

# prob = OptimizationProblem(sys, x, [a => 1.0, b => 1.0],
#     grad = true, hess = true, cons_j = true, cons_h = true)

x00 = [0.01,0.8,0.0]
    xx0 = [x[1] => 0.01, 
        x[2] => -0.8, 
        x[3] => 0.0]

prob = OptimizationProblem(sys, xx0, [a => 1.0, b => 1.0];lb=lbs,ub=ubs,
    grad = true, hess = true, cons_j = true, cons_h = true)


# loss = (1.0 - x[1])^2 + x[3]^2
# @named sys = OptimizationSystem(loss, x, []; constraints = cons)
# sys = structural_simplify(sys)
# prob = OptimizationProblem(sys, [x[1] => 0.01, x[2] => -0.8, x[3] => 0.0];lb=lbs,ub=ubs,
#     grad = true, hess = true, cons_j = true, cons_h = true)


sol = solve(prob, IPNewton())



