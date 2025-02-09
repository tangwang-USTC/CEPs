# using JuMP, Dualization

# model = Model()
# # ... build model ...
# dual_model = dualize(model)

using JuMP, Dualization, SCS
model = Model(dual_optimizer(SCS.Optimizer))
# ... build model ...
JuMP.optimize!(model)  # Solves the dual instead of the primal

