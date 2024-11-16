
using Plots,DataFrames,CSV,Format,OhMyREPL

pathroot = "C:/Users/Administrator/.julia/pkgGithub/CPEs"

include(joinpath(pathroot,"Mathematics/maths.jl"))
include(joinpath(pathroot,"src/ES/ESs.jl"))
include(joinpath(pathroot,"test/run_collisions/algorithm/modules.jl"))
include(joinpath(pathroot,"test/run_collisions/paras_alg_optim.jl"))

datatype = BigFloat
# datatype = Float64
is_change_datatype = true
is_change_datatype = false

rtol_OrjL = 1e-14
RDnuT = -1e-6

is_re_seed = true
# is_re_seed = false
is_show_nuTi = false
is_optim =  true

        
is_C = true 
# is_C = false 

is_norm_uhL = true
# is_norm_uhL = false
        
    is_Jacobian = true
    # is_Jacobian = false

nModL = 3
njL = 3 * nModL
is_anasys_L = false

is_L_even = 5
println("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
if is_L_even == 1
    L = 0
    include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
    L = 2
    include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
    L = 4
    include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
    L = 6
    include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
    L = 8
    include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
elseif is_L_even == 0
    L = 1
    include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
    L = 3
    include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
    L = 5
    include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
    L = 7
    include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
    L = 9
    include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
else
    L = 0
    include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
    L = 1
    include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
    L = 2
    include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
    L = 3
    include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
    L = 4
    include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
    L = 5
    include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
    L = 6
    include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
    L = 7
    include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
    L = 8
    include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
    L = 9
    include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
end
