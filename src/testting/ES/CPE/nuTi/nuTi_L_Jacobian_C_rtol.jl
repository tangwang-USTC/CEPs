
using Plots,DataFrames,CSV,Format,OhMyREPL

pathroot = "C:/Users/Administrator/.julia/pkgGithub/CPEs"

include(joinpath(pathroot,"Mathematics/maths.jl"))
include(joinpath(pathroot,"src/ES/ESs.jl"))
include(joinpath(pathroot,"test/run_collisions/algorithm/modules.jl"))
include(joinpath(pathroot,"test/run_collisions/paras_alg_optim.jl"))

is_change_datatype = false

rtol_OrjL = 1e-14
RDnuT = 0e-16

is_re_seed = true
is_re_seed = false
is_show_nuTi = true
is_optim =  true

nModL = 3

L = 3 + 0
include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/nuTi_L_Jacobian_C_rtol_kerl.jl"))
