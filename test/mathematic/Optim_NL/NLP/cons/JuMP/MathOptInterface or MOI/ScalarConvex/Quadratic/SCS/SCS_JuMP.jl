using JuMP, SCS       #  linear programs, second-order cone programs, semidefinite programs, exponential cone programs, and power cone programs.

items  = [:Gold, :Silver, :Bronze]

values = Dict(:Gold => 5.0,  :Silver => 3.0,  :Bronze => 1.0)
weight = Dict(:Gold => 2.0,  :Silver => 1.5,  :Bronze => 0.3)

model = Model(SCS.Optimizer)

@variable(model, 0 <= take[items] <= 1)  # Define a variable for each item

@objective(model, Max, sum(values[item] * take[item] for item in items))

@constraint(model, sum(weight[item] * take[item] for item in items) <= 3)

JuMP.optimize!(model)

println(value.(take))

# 1-dimensional DenseAxisArray{Float64,1,...} with index sets:
#     Dimension 1, Symbol[:Gold, :Silver, :Bronze]
# And data, a 3-element Array{Float64,1}:
#  1.0000002002226671
#  0.4666659513182934
#  1.0000007732744878