# using KahanSummation

using Plots,DataFrames,CSV,Format,OhMyREPL

pathroot = "C:/Users/Administrator/.julia/pkgGithub/CPEs"

include(joinpath(pathroot,"Mathematics/maths.jl"))
include(joinpath(pathroot,"src/ES/ESs.jl"))
include(joinpath(pathroot,"test/run_collisions/algorithm/modules.jl"))
include(joinpath(pathroot,"test/run_collisions/paras_alg_optim.jl"))


L = 0
nModL0 = 3
maxIterKing = 500
# is_C = false                                             # maybe be conservative when `RDnuT` is moderate
# is_C = true                                              # maybe not be conservative when `RDnuT` is bigger enough

show_trace = false
# show_trace = true

(factor, factor_abbr) = (LeastSquaresOptim.QR(), :QR)     # `=QR(), default`  # More stability
# factor = LeastSquaresOptim.Cholesky()
# factor = LeastSquaresOptim.LSMR()                       # 最差

RDnuT = -1e-12
# is_Jacobian = true                                       # maybe not be conservative when `RDnuT` is bigger enough
is_re_seed = false
is_re_seed = true

# 先格式预测，再非守恒优化，最后再守恒优化。
# 同时测试QR与Cholesky，同时比较is_Jacobian与否，择优而行。
# 若非理想，则减小步长，重新寻优。

if is_re_seed
        naiL0 = rand(nModL0)
        naiL0 /= sum_kbn(naiL0)
        # naiL0 /= sum_kbn(naiL0)
        # naiL00 = naiL0 / sum_kbn(naiL0)
        # sum_kbn(naiL00)-1, sum_kbn(naiL0)-1
        
        
        vthiL0 = rand(nModL0)
        vthiL0 /= sum_kbn(vthiL0)
        vthiL0 *= nModL0
        
        uaiL0 = randn(nModL0)
        uaiL0 /= (maximum(abs.(uaiL0)) * 5)
        # uaiL0 *= 0.0
end

njL0 = 3 * nModL0
njML0 = njL0 + 0
jvecL0 = 0:2:2(njL0-1) |> Vector{Int}

mathtype = :Exact       # [:Exact, :Taylor0, :Taylor1, :TaylorInf]
is_renorm = true
# is_renorm = false
Mhj0 = zeros(njML0)
MhsKMM0!(Mhj0,jvecL0,naiL0,uaiL0,vthiL0,nModL0;is_renorm=is_renorm,rtol_OrjL=rtol_OrjL,mathtype=mathtype)

# println()
# Msnnt = zeros(njML0)
# Msnnt = MsnntL2fL0(Msnnt,njML0,L,naiL0,uaiL0,vthiL0,nModL0;is_renorm=is_renorm)


Nspan_optim_nuTi = [1.0,1.0, 1.0]
DMh024 = [1.0,1.0, 1.0]
rtol_OrjL = 1e-15
atol_Mh = 1e-15
rtol_Mh = 1e-15

println("........................................................................")
is_C = false   
include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/nuTi_L0_kerl.jl"))


is_C = true 
include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/nuTi_L0_kerl.jl"))


