using ModelingToolkit, SparseArrays, Test, Optimization, OptimizationOptimJL,
      OptimizationMOI, Ipopt, AmplNLWriter, Ipopt_jll
using ModelingToolkit: get_metadata

@variables x y z
@parameters a b

# loss = (a - x)^2 + b * z^2

loss = (a - x)^2
loss += b * z^2

function lossfun(x::Num,y::Num,a::Num,b::Num)
    
    loss = (a - x)^2
    @show typeof(loss), loss
    loss += b * z^2
    @show typeof(loss), loss
    return loss
end

loss = lossfun(x,y,a,b)

cons = [1.0 ~ x^2 + y^2
        z ~ y - x^2
        z^2 + y^2 ≲ 1.0]
@named sys = OptimizationSystem(loss, [x, y, z], [a, b], constraints = cons)
sys = structural_simplify(sys)
prob = OptimizationProblem(sys, [x => 0.0, y => 0.0, z => 0.0], [a => 1.0, b => 1.0],
    grad = true, hess = true, cons_j = true, cons_h = true)
sol = solve(prob, IPNewton())



