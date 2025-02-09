using JuMP
import MiniZinc

ENV["JULIA_LIBMINIZINC_DIR"] = "C:\\Program Files\\MiniZinc"

model = Model(() -> MiniZinc.Optimizer{Float64}("highs"))

@variable(model, 1 <= x[1:3] <= 3, Int)
@constraint(model, x in MOI.AllDifferent(3))
@objective(model, Max, sum(i * x[i] for i in 1:3))

optimize!(model)

@show value.(x)