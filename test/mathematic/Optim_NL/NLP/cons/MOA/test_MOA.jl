

using Test

import HiGHS
import Ipopt
import MultiObjectiveAlgorithms as MOA
import MultiObjectiveAlgorithms: MOI

f = MOI.OptimizerWithAttributes(
    () -> MOA.Optimizer(HiGHS.Optimizer),
    MOA.Algorithm() => MOA.Dichotomy(),
)

model = MOI.instantiate(f)
MOI.set(model, MOI.Silent(), true)

MOI.Utilities.loadfromstring!(
    model,
    """
    variables: x, y
    minobjective: [2 * x + y + 1, x + 3 * y]
    c1: x + y >= 1.0
    c2: 0.5 * x + y >= 0.75
    c3: x >= 0.0
    c4: y >= 0.25
    """,
)

x = MOI.get(model, MOI.VariableIndex, "x")
y = MOI.get(model, MOI.VariableIndex, "y")

MOI.optimize!(model)

@test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
@test MOI.get(model, MOI.ResultCount()) == 3
X = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.25]]
Y = [[2.0, 3.0], [2.5, 2.0], [3.25, 1.75]]
for i in 1:3
    @test MOI.get(model, MOI.PrimalStatus(i)) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.DualStatus(i)) == MOI.NO_SOLUTION
    @test MOI.get(model, MOI.ObjectiveValue(i)) == Y[i]
    @test MOI.get(model, MOI.VariablePrimal(i), x) == X[i][1]
    @test MOI.get(model, MOI.VariablePrimal(i), y) == X[i][2]
end
@test MOI.get(model, MOI.ObjectiveBound()) == [2.0, 1.75]


