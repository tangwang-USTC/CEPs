
using Plots,DataFrames,CSV,Format,OhMyREPL

pathroot = "C:/Users/Administrator/.julia/pkgGithub/CPEs"

include(joinpath(pathroot,"Mathematics/maths.jl"))
include(joinpath(pathroot,"src/ES/ESs.jl"))
include(joinpath(pathroot,"test/run_collisions/algorithm/modules.jl"))
include(joinpath(pathroot,"test/run_collisions/paras_alg_optim.jl"))

is_change_datatype = true
is_change_datatype = false

rtol_OrjL = 1e-14
RDnuT = 1e-8

is_re_seed = true
is_re_seed = false
is_show_nuTi = true
is_optim =  true

is_norm_uhL = true
# is_norm_uhL = false
nModL = 3

njL = 3 * nModL
L = 2 + 0
include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/nuTi_L_Jacobian_C_rtol_kerl.jl"))
