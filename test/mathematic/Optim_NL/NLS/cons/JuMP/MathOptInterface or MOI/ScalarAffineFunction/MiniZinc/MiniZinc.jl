import MiniZinc

import MathOptInterface as MOI

ENV["JULIA_LIBMINIZINC_DIR"] = "C:\\Users\\Administrator\\.julia\\packages\\MiniZinc"

function main()

    model = MOI.Utilities.CachingOptimizer(
        MiniZinc.Model{Int}(),
        MiniZinc.Optimizer{Int}("chuffed"),
    )
    # xᵢ ∈ {1, 2, 3} ∀i=1,2,3
    x = MOI.add_variables(model, 3)
    MOI.add_constraint.(model, x, MOI.Interval(1, 3))
    MOI.add_constraint.(model, x, MOI.Integer())
    # zⱼ ∈ {0, 1}    ∀j=1,2
    z = MOI.add_variables(model, 2)
    MOI.add_constraint.(model, z, MOI.ZeroOne())
    # z₁ <-> x₁ != x₂
    MOI.add_constraint(
        model,
        MOI.VectorOfVariables([z[1], x[1], x[2]]),
        MOI.Reified(MOI.AllDifferent(2)),
    )
    # z₂ <-> x₂ != x₃
    MOI.add_constraint(
        model,
        MOI.VectorOfVariables([z[2], x[2], x[3]]),
        MOI.Reified(MOI.AllDifferent(2)),
    )
    # z₁ + z₂ = 1
    MOI.add_constraint(model, 1 * z[1] + x[2], MOI.EqualTo(1))
    MOI.optimize!(model)
    x_star = MOI.get(model, MOI.VariablePrimal(), x)
    z_star = MOI.get(model, MOI.VariablePrimal(), z)
    return x_star, z_star
end

main()

