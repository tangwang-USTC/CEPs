using JuMP, Ipopt

function solve_constrained_least_squares_regression(A::Matrix, b::Vector)
           m, n = size(A)
           model = Model(Ipopt.Optimizer)
           set_silent(model)
           @variable(model, x[1:n])
           @variable(model, residuals[1:m])
           @constraint(model, residuals == A * x - b)
           @constraint(model, sum(x) == 1)
           @objective(model, Min, sum(residuals.^2))
           optimize!(model)
           return value.(x)
       end

A, b = rand(10, 3), rand(10);

x = solve_constrained_least_squares_regression(A, b)
