# using KahanSummation

using Plots,DataFrames,CSV,Format,OhMyREPL

pathroot = "C:/Users/Administrator/.julia/pkgGithub/CPEs"

include(joinpath(pathroot,"Mathematics/maths.jl"))
include(joinpath(pathroot,"src/ES/ESs.jl"))
include(joinpath(pathroot,"test/run_collisions/algorithm/modules.jl"))
include(joinpath(pathroot,"test/run_collisions/paras_alg_optim.jl"))


L = 7           # L ≥ 4 ? re-renormalization: (uhh^L)
nModL = 2
maxIterKing = 500
# is_C = false                                             # maybe be conservative when `RDnuT` is moderate
# is_C = true                                              # maybe not be conservative when `RDnuT` is bigger enough

show_trace = false
# show_trace = true

(factor, factor_abbr) = (LeastSquaresOptim.QR(), :QR)     # `=QR(), default`  # More stability
# factor = LeastSquaresOptim.Cholesky()
# factor = LeastSquaresOptim.LSMR()                       # 最差

RDnuT = 1e-10
# is_Jacobian = true                                       # maybe not be conservative when `RDnuT` is bigger enough
is_re_seed = false
# is_re_seed = true

# 对 `(uhh^L)` 再次归一化
# 先格式预测，再非守恒优化，最后再守恒优化。
# 同时测试QR与Cholesky，同时比较is_Jacobian与否，择优而行。
# 若非理想，则减小步长，重新寻优。

if is_re_seed
        naiL = rand(nModL)
        naiL /= sum(naiL)
        # naiL /= sum_kbn(naiL)
        # naiL = naiL / sum_kbn(naiL)
        # sum(naiL)-1, sum(naiL)-1
        
        
        vthiL = rand(nModL)
        vthiL /= sum(vthiL)
        vthiL *= nModL
        
        uaiL = randn(nModL)
        uaiL /= (maximum(abs.(uaiL)) * 5)
        # uaiL *= 0.0
end

njL = 3 * nModL
njML = njL + 0
jvecL = 0:2:2(njL-1) |> Vector{Int}
jvecL .+= L

mathtype = :Exact       # [:Exact, :Taylor0, :Taylor1, :TaylorInf]
is_renorm = true
# is_renorm = false

uhLN = uhLNorm(naiL,uaiL,L)
uhLNL = uhLN .^L
@show uhLNL

MhjL = zeros(njML)
MhsKMM!(MhjL,jvecL,L,naiL,uaiL,vthiL,nModL;is_renorm=is_renorm,mathtype=mathtype)

# println()
# Msnnt = zeros(njML)
# Msnnt = MsnntL2fL(Msnnt,njML,L,naiL,uaiL,vthiL,nModL;is_renorm=is_renorm)


Nspan_optim_nuTi = [1.0,1.0, 1.0]
DMh024 = [1.0,1.0, 1.0]
rtol_OrjL = 1e-15
atol_Mh = 1e-15
rtol_Mh = 1e-15

is_show_Dc = false
println("........................................................................")
is_C = false   
is_Jacobian = false
include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/nuTi_L_kerlkerl.jl"))

is_Jacobian = true
include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/nuTi_L_kerlkerl.jl"))


is_C = true 
is_show_Dc = true
is_Jacobian = true
include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/nuTi_L_kerlkerl.jl"))

is_Jacobian = false
include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/nuTi_L_kerlkerl.jl"))


