# using KahanSummation

using Plots,DataFrames,CSV,Format,OhMyREPL

pathroot = "C:/Users/Administrator/.julia/pkgGithub/CPEs"

include(joinpath(pathroot,"Mathematics/maths.jl"))
include(joinpath(pathroot,"src/ES/ESs.jl"))
include(joinpath(pathroot,"test/run_collisions/algorithm/modules.jl"))
include(joinpath(pathroot,"test/run_collisions/paras_alg_optim.jl"))


L = 0
nModL0 = 3

show_trace = false
# show_trace = true

(factor, factor_abbr) = (LeastSquaresOptim.QR(), :QR)     # `=QR(), default`  # More stability
# factor = LeastSquaresOptim.Cholesky()
# factor = LeastSquaresOptim.LSMR()                       # 最差
maxIterKing = 500
is_C = false                                             # maybe be conservative when `RDnuT` is moderate
# is_C = true                                              # maybe not be conservative when `RDnuT` is bigger enough

RDnuT = - 1e-5
# is_Jacobian = true                                       # maybe not be conservative when `RDnuT` is bigger enough
is_re_seed = false
# is_re_seed = true

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


if 1 == 1
        Nspan_optim_nuTi = [1.0,1.0, 1.0]
        DMh024 = [1.0,1.0, 1.0]
        rtol_OrjL = 1e-15
        atol_Mh = 1e-15
        rtol_Mh = 1e-15
        
        scales = 1.0 + RDnuT
        naiLt0,uaiLt0,vthiLt0 = scales * naiL0,scales * uaiL0,scales * vthiL0
        Mhr0s = naiLt0

        Mhr2s = naiLt0 .* (vthiLt0.^2 + CjLk(2,1) * uaiLt0.^2)
        # @show Mhj0[2] - sum_kbn(Mhr2s)
        # @show sum_kbn(Mhr2s), Mhr2s

        Or40 = zeros(nModL0)
        Orj0Nb!(Or40,(uaiLt0./vthiLt0).^2,4,2,nModL0;rtol_OrjL=rtol_OrjL)
        Mhr4s = (naiLt0 .* (vthiLt0).^4) .* (1 .+ Or40)
        # @show Mhj0[3] - sum_kbn(Mhr4s)
        uhh2 = 1.5 * (2.5 * ((Mhr2s ./ Mhr0s).^2 - (Mhr4s ./ Mhr0s))).^0.5

        println("//////////////////////////////")
        # @show Mhr0s[nModL0],Mhr2s[nModL0],Mhr4s[nModL0] 

        is_Jacobian = false
        if is_C
                xssr, is_converged, xfit, niter = optimMhjLC!(naiLt0,uaiLt0,vthiLt0,Mhj0,L,nModL0,NL_solve,DMh024;
                        rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh,
                        optimizer=optimizer,factor=factor,autodiff=autodiff,
                        is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                        p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,NL_solve_method=NL_solve_method,
                        Nspan_optim_nuTi=Nspan_optim_nuTi)
        else
                xssr, is_converged, xfit, niter = optimMhjL!(naiLt0,uaiLt0,vthiLt0,Mhj0,L,nModL0,NL_solve,DMh024;
                        rtol_OrjL=rtol_OrjL,
                        optimizer=optimizer,factor=factor,autodiff=autodiff,
                        is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                        p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,NL_solve_method=NL_solve_method,
                        Nspan_optim_nuTi=Nspan_optim_nuTi)
        end
        
        if is_converged
                printstyled("(niter, xssr,factor)=",(niter, fmt2(xssr),factor);color=:green)
        else
                printstyled("(niter, xssr,factor)=",(niter, fmt2(xssr),factor);color=:red)
        end
        # if is_converged
            naio = xfit[1:3:end]
            uaio = xfit[2:3:end]
        #     if uaiLt0[nModL0] < 0.0
        #         uaio[nModL0] *= - 1
        #     end
            vthio = xfit[3:3:end]
            Mhj0o = zeros(njML0)
            MhsKMM0!(Mhj0o,jvecL0,naio,uaio,vthio,nModL0;is_renorm=is_renorm,rtol_OrjL=rtol_OrjL,mathtype=mathtype)
            
        # end
        println()
        @show is_C, is_re_seed, is_Jacobian, RDnuT
        @show fmt2(sum_kbn(naio) - 1.0)
        @show fmt2(sum_kbn(naio .* uaio .^2) / sum_kbn(naiL0 .* uaiL0 .^2) - 1.0)
        @show fmt2.(Mhj0o[1:3nModL0] ./ Mhj0[1:3nModL0] .- 1)
        @show fmt2.(naio ./ naiL0 .- 1)
        @show fmt2.(abs.(uaio ./ uaiL0) .- 1)
        @show fmt2.(vthio ./ vthiL0 .- 1)  
        nuToptMatric = [naiL0 naio uaiL0 uaio vthiL0 vthio uaiL0./vthiL0 uaio./vthio]  
        nuToptName = ["naiL0", "naio", "uaiL0", "uaio", "vthiL0", "vthio", "uhhL0", "uhho"]
        nuTopt = DataFrame(nuToptMatric,:auto)
        rename!(nuTopt,nuToptName)  
        @show nuTopt




        is_Jacobian = true
        if is_C
                xssr, is_converged, xfit, niter = optimMhjLC!(naiLt0,uaiLt0,vthiLt0,Mhj0,L,nModL0,NL_solve,DMh024;
                        rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh,
                        optimizer=optimizer,factor=factor,autodiff=autodiff,
                        is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                        p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,NL_solve_method=NL_solve_method,
                        Nspan_optim_nuTi=Nspan_optim_nuTi)
        else
                xssr, is_converged, xfit, niter = optimMhjL!(naiLt0,uaiLt0,vthiLt0,Mhj0,L,nModL0,NL_solve,DMh024;
                        rtol_OrjL=rtol_OrjL,
                        optimizer=optimizer,factor=factor,autodiff=autodiff,
                        is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                        p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,NL_solve_method=NL_solve_method,
                        Nspan_optim_nuTi=Nspan_optim_nuTi)
        end
        
        if is_converged
                printstyled("(niter, xssr,factor)=",(niter, fmt2(xssr),factor);color=:green)
        else
                printstyled("(niter, xssr,factor)=",(niter, fmt2(xssr),factor);color=:red)
        end
        # if is_converged
            naio = xfit[1:3:end]
            uaio = xfit[2:3:end]
        #     if uaiLt0[nModL0] < 0.0
        #         uaio[nModL0] *= - 1
        #     end
            vthio = xfit[3:3:end]
            Mhj0o = zeros(njML0)
            MhsKMM0!(Mhj0o,jvecL0,naio,uaio,vthio,nModL0;is_renorm=is_renorm,rtol_OrjL=rtol_OrjL,mathtype=mathtype)
            
        # end
        println()
        @show is_C, is_re_seed, is_Jacobian, RDnuT
        @show fmt2(sum_kbn(naio) - 1.0)
        @show fmt2(sum_kbn(naio .* uaio .^2) / sum_kbn(naiL0 .* uaiL0 .^2) - 1.0)
        @show fmt2.(Mhj0o[1:3nModL0] ./ Mhj0[1:3nModL0] .- 1)
        @show fmt2.(naio ./ naiL0 .- 1)
        @show fmt2.(abs.(uaio ./ uaiL0) .- 1)
        @show fmt2.(vthio ./ vthiL0 .- 1)  
        nuToptMatric = [naiL0 naio uaiL0 uaio vthiL0 vthio uaiL0./vthiL0 uaio./vthio]  
        nuToptName = ["naiL0", "naio", "uaiL0", "uaio", "vthiL0", "vthio", "uhhL0", "uhho"]
        nuTopt = DataFrame(nuToptMatric,:auto)
        rename!(nuTopt,nuToptName)  
end
