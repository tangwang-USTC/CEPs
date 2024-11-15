
using Plots,DataFrames,CSV,Format,OhMyREPL

pathroot = "C:/Users/Administrator/.julia/pkgGithub/CPEs"

include(joinpath(pathroot,"Mathematics/maths.jl"))
include(joinpath(pathroot,"src/ES/ESs.jl"))
include(joinpath(pathroot,"test/run_collisions/algorithm/modules.jl"))
include(joinpath(pathroot,"test/run_collisions/paras_alg_optim.jl"))

is_change_datatype = false
rtol_OrjL = 1e-14
RDnuT = -1e-8

is_re_seed = true
is_re_seed = false
is_show_nuTi = false
is_optim =  true

nModL = 3

L = 0
include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/nuTi_L_Jacobian_C_rtol_kerl.jl"))

is_re_seed = false
L = 2
include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/nuTi_L_Jacobian_C_rtol_kerl.jl"))
L = 4
include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/nuTi_L_Jacobian_C_rtol_kerl.jl"))
L = 6
include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/nuTi_L_Jacobian_C_rtol_kerl.jl"))
L = 8
include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/nuTi_L_Jacobian_C_rtol_kerl.jl"))
L = 10
is_show_nuTi = true
include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/nuTi_L_Jacobian_C_rtol_kerl.jl"))
# L = 12
# include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/nuTi_L_Jacobian_C_rtol_kerl.jl"))
# L = 14
# include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/nuTi_L_Jacobian_C_rtol_kerl.jl"))
# L = 16
# include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/nuTi_L_Jacobian_C_rtol_kerl.jl"))
# L = 18
# include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/nuTi_L_Jacobian_C_rtol_kerl.jl"))
# L = 20
# include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/nuTi_L_Jacobian_C_rtol_kerl.jl"))
# L = 22
# include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/nuTi_L_Jacobian_C_rtol_kerl.jl"))
# L = 24
# include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/nuTi_L_Jacobian_C_rtol_kerl.jl"))
# L = 26
# include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/nuTi_L_Jacobian_C_rtol_kerl.jl"))
# L = 28
# is_show_nuTi = true
# include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/nuTi_L_Jacobian_C_rtol_kerl.jl"))
