using JuMP
import EAGO

# Build model using EAGO's optimizer
model = Model(EAGO.Optimizer)

# Define bounded variables
xL = [10.0, 0.0, 0.0, 0.0, 0.0, 85.0, 90.0, 3.0, 1.2, 145.0]
xU = [2000.0, 16000.0, 120.0, 5000.0, 2000.0, 93.0, 95.0, 12.0, 4.0, 162.0]

@variable(model, xL[i] <= x[i=1:10] <= xU[i])
# Define nonlinear constraints
@NLconstraints(model, begin
    -x[1]*(1.12 + 0.13167*x[8] - 0.00667*(x[8])^2) + x[4] == 0.0
    -0.001*x[4]*x[9]*x[6]/(98.0 - x[6]) + x[3] == 0.0
    -(1.098*x[8] - 0.038*(x[8])^2) - 0.325*x[6] + x[7] == 57.425
    -(x[2] + x[5])/x[1] + x[8] == 0.0
end)

# Define linear constraints
@constraints(model, begin
    -x[1] + 1.22*x[4] - x[5] == 0.0
    x[9] + 0.222*x[10] == 35.82
    -3.0*x[7] + x[10] == -133.0
end)

# Define nonlinear objective
@NLobjective(
    model, 
    Max,
    0.063*x[4]*x[7] - 5.04*x[1] - 0.035*x[2] - 10*x[3] - 3.36*x[5],
)

# Solve the optimization problem
JuMP.optimize!(model)


solx = JuMP.value.(x)
global_obj_value = objective_value(model)
print("Global solution at y*=$solx with a value of f*=$global_obj_value")

