
using ModelingToolkit

@variables x y z
@parameters a b c

obj = a * (y - x) + x * (b - z) - y
obj +=  x * y - c * z

cons = [x^2 + y^2 ≲ 1]
@named os = OptimizationSystem(obj, [x, y, z], [a, b, c]; constraints = cons)

