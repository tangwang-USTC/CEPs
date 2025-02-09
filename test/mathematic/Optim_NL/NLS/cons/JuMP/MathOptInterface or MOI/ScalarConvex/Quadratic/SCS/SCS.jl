using Convex, SCS

items  = [:Gold, :Silver, :Bronze]
values = [5.0, 3.0, 1.0]
weights = [2.0, 1.5, 0.3]

# Define a variable of size 3, each index representing an item
x = Variable(3)
p = maximize(x' * values, 0 <= x, x <= 1, x' * weights <= 3)

solve!(p, SCS.Optimizer)

println([items x.value])

# [:Gold 0.9999971880377178
#  :Silver 0.46667637765641057
#  :Bronze 0.9999998036351865]

