using Flux, Metalhead, Zygote, Optimisers

model = Metalhead.ResNet(18) |> gpu  # define a model to train
image = rand(Float32, 224, 224, 3, 1) |> gpu;  # dummy data
@show sum(model(image));  # dummy loss function

rule = Optimisers.Adam()  # use the Adam optimiser with its default settings
state_tree = Optimisers.setup(rule, model);  # initialise this optimiser's momentum etc.

∇model, _ = gradient(model, image) do m, x  # calculate the gradients
  sum(m(x))
end;

state_tree, model = Optimisers.update(state_tree, model, ∇model);
@show sum(model(image));  # reduced

