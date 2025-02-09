using JuMP, MathOptAI, Flux

predictor = Flux.Chain(
           Flux.Dense(28^2 => 32, Flux.sigmoid),
           Flux.Dense(32 => 10),
           Flux.softmax,
       );

#= Train the Flux model. Code not shown for simplicity =#

model = JuMP.Model();

JuMP.@variable(model, 0 <= x[1:28^2] <= 1);

y, formulation = MathOptAI.add_predictor(model, predictor, x);

y
