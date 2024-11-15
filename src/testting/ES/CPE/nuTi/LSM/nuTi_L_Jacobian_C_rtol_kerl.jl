

maxIterKing = 500
# is_C = false                                             # maybe be conservative when `RDnuT` is moderate
# is_C = true                                              # maybe not be conservative when `RDnuT` is bigger enough

show_trace = false
# show_trace = true

(factor, factor_abbr) = (LeastSquaresOptim.QR(), :QR)     # `=QR(), default`  # More stability
# factor = LeastSquaresOptim.Cholesky()
# factor = LeastSquaresOptim.LSMR()                       # 最差

# is_Jacobian = true                                       # maybe not be conservative when `RDnuT` is bigger enough
# is_re_seed = false
# is_re_seed = true

is_renorm = true
# is_renorm = false


is_plot_MhjL = false
# is_plot_MhjL = true
is_logplot_MhjL = true

# 对 `(uhh^L)` 再次归一化
# 先格式预测，再非守恒优化，最后再守恒优化。
# 同时测试QR与Cholesky，同时比较is_Jacobian与否，择优而行。
# 若非理想，则减小步长，重新寻优。
# `nMod` 越大，优化的精度越低。

if is_change_datatype 
        include(joinpath(pathroot,"Mathematics/consts_datatype.jl"))
end

if is_re_seed
        naiL = rand(datatype,nModL)
        naiL /= sum_kbn(naiL)
        # naiL /= sum_kbn(naiL)
        # naiL = naiL / sum_kbn(naiL)
        # sum_kbn(naiL)-1, sum_kbn(naiL)-1
        
        
        vthiL = rand(datatype,nModL)
        vthiL /= sum_kbn(vthiL)
        vthiL *= nModL
        
        uaiL = randn(datatype,nModL)
        uaiL /= (maximum(abs.(uaiL)) * 5)
        # uaiL *= 0.0
end
# if datatype ≠ Float64
#         naiL = datatype.(naiL)
#         uaiL = datatype.(uaiL)
#         vthiL = datatype.(vthiL)
# end
if datatype ≠ Float64
        rtol_OrjL = datatype(rtol_OrjL)
end

njML = njL + 0
jvecL = 0:2:2(njL-1) |> Vector{Int}
jvecL .+= L

mathtype = :Exact       # [:Exact, :Taylor0, :Taylor1, :TaylorInf]

uhLN = uhLNorm(naiL,uaiL,L)
# uhLN = maximum(abs.(uaiL))

# uhLN *= 1.3
# uhLN = 1.0 |> datatype

uhLNL = uhLN .^L

MhjL = zeros(datatype,njML)
MhsKMM!(MhjL,jvecL,L,naiL,uaiL,vthiL,uhLN,nModL;is_renorm=is_renorm,is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL,mathtype=mathtype)

Diff_MhjL = Float64.(diff(MhjL[2:end]))
rate_MhjL = Float64.(maximum(abs.(MhjL)) / minimum(abs.(MhjL)))

if is_show_nuTi 
    @show Float64.(uhLN), Float64.(uhLNL), rate_MhjL
    nuTMatrix = [naiL uaiL vthiL uaiL./vthiL]  
    nuTName = ["naiL", "uaiL", "vthiL", "uhhL"]
    nuT = DataFrame(Float64.(nuTMatrix),:auto)
    rename!(nuT,nuTName)  
    @show nuT
end

if is_plot_MhjL
        label = string("sign,RMhjL=",fmtf2(rate_MhjL))
        pDiff_MhjL = plot(sign.(Diff_MhjL),label=label)
        ylabel!("diff(MhjL)")

        MhjL_sign = sign.(MhjL[2:end])
        MhjL_log = log.(abs.(MhjL[2:end]))
        label = string("jM,L=",(jvecL[end],L))
        pMhjLlog = plot(MhjL_sign .* MhjL_log,label=label)
        # xlabel!("j")
        ylabel!("log(|MhjL|)")

        pDMhjLlog = plot(diff(MhjL_log),label=label)
        xlabel!("j")
        ylabel!("D(log(|MhjL|))")

        label = string("jM,L=",(jvecL[end],L))
        pMhjL = plot(MhjL[2:end],label=label)
        xlabel!("j")
        ylabel!("MhjL")
        display(plot(pDiff_MhjL, pMhjLlog, pMhjL, pDMhjLlog,layout=(2,2)))

        @show fmtf2.(MhjL)
        @show fmtf2.(MhjL_log)
        @show fmtf2.(diff(MhjL_log))
    
end
# println()
# Msnnt = zeros(njML)
# Msnnt = MsnntL2fL(Msnnt,njML,L,naiL,uaiL,vthiL,nModL;is_renorm=is_renorm)

if is_optim 
        Nspan_optim_nuTi = [1.0,1.0, 1.0] |> Vector{datatype}
        DMh024 = [1.0,1.0, 1.0] |> Vector{datatype}
        rtol_OrjL = 1e-15 |> datatype
        atol_Mh = 1e-15 |> datatype
        rtol_Mh = 1e-15 |> datatype
        
        is_show_Dc = false
        println("........................................................................")

        # is_C = false   

        # is_Jacobian = false
        # include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_optim_kerl.jl"))
        
        # is_Jacobian = true
        # # is_show_Dc = true
        # include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_optim_kerl.jl"))
        
        
        is_C = true 
        
        is_show_Dc = false
        is_Jacobian = false
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_optim_kerl.jl"))
        
        # is_show_Dc = true
        # is_Jacobian = true
        # include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_optim_kerl.jl"))
        
end
