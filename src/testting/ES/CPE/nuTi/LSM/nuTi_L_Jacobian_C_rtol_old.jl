
using Plots,DataFrames,CSV,Format,OhMyREPL

pathroot = "C:/Users/Administrator/.julia/pkgGithub/CPEs"

include(joinpath(pathroot,"Mathematics/maths.jl"))
include(joinpath(pathroot,"src/ES/ESs.jl"))
include(joinpath(pathroot,"test/run_collisions/algorithm/modules.jl"))
include(joinpath(pathroot,"test/run_collisions/paras_alg_optim.jl"))

datatype = BigFloat
datatype = Float64
is_change_datatype = true
is_change_datatype = false

rtol_OrjL = 1e-14
RDnuT = -0e-8

is_re_seed = true
# is_re_seed = false
is_show_nuTi = false
is_optim =  false
is_optim =  true

is_norm_uhL = true
# is_norm_uhL = false
nModL = 3         # when `L` is odd number, the optimization is more challenge!
if nModL == 3
    naiLok = datatype.([0.4429805367270422470174924231902531025012880842972989134553914903140590230558192
            0.2266138115283708717468598063515828621194924776557254115578782704094342418434013
            0.3304056517445868812356477704581640353792194380469756749867302392765067351007709])
    uaiLok = datatype.([0.05034178675563209889943384505709961613170249352291586568335469388449438035259866
            -0.00330104437197134090547557153815187183884883604346171135816840427024951454698761
            0.2000000000000000000000000000000000000000000000000000000000000000000000000000004])
    vthiLok = datatype.([2.172107436802767216842088574135748210446180805776723195502077652793266120892492
            0.3964075780707377171697420092387201548468603708624627768275445691579439979825734
            0.4314849851264950659881694166255316347069588233608140276703777780487898811249299])
end

njL = 3 * nModL
L = 1 + 0
include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))

is_re_seed = false
# L = 1 + 2
# include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
L = 1 + 4
include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
L = 1 + 6
include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
L = 1 + 8
is_show_nuTi = true
include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
# L = 1 + 10
# include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
# L = 1 + 12
# include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
# L = 1 + 14
# include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
# L = 1 + 16
# include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
# L = 1 + 18
# include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
# L = 1 + 20
# include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
# L = 1 + 22
# include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
# L = 1 + 24
# include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
# L = 1 + 26
# include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
# L = 1 + 28
# is_show_nuTi = true
# include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
