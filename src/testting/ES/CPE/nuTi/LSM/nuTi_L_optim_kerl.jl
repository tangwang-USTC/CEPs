

if 1 == 1
    scales = 1.0 + RDnuT
    naiLt0,uaiLt0,vthiLt0 = scales * naiL,scales * uaiL,scales * vthiL
    Mhr0s = naiLt0

    # Mhr2s = naiLt0 .* (vthiLt0.^2 + CjLk(2,1) * uaiLt0.^2)
    # # @show MhjL[2] - sum_kbn(Mhr2s)
    # # @show sum_kbn(Mhr2s), Mhr2s

    # Or40 = zeros(datatype,nModL)
    # Orj0Nb!(Or40,(uaiLt0./vthiLt0).^2,4,2,nModL;rtol_OrjL=rtol_OrjL)
    # Mhr4s = (naiLt0 .* (vthiLt0).^4) .* (1 .+ Or40)
    # # @show MhjL[3] - sum_kbn(Mhr4s)
    # uhh2 = 1.5 * (2.5 * ((Mhr2s ./ Mhr0s).^2 - (Mhr4s ./ Mhr0s))).^0.5

    # println("//////////////////////////////")
    # @show Mhr0s[nModL],Mhr2s[nModL],Mhr4s[nModL] 
    if is_C
            xssr, is_converged, xfit, niter = optimMhjLC!(naiLt0,uaiLt0,vthiLt0,uhLN,MhjL,L,nModL,DMh024;
                    atol_Mh=atol_Mh,rtol_Mh=rtol_Mh,NL_solve=NL_solve,
                    Optlibary=Optlibary,NL_solve_method=NL_solve_method,optimizer=optimizer,
                    linsolve=linsolve,linesearch=linesearch,preconditioner=preconditioner,ADtype=ADtype, 
                    is_norm_uhL=is_norm_uhL,is_Jacobian=is_Jacobian,is_Hessian=is_Hessian,is_AD=is_AD,
                    is_constraint=is_constraint,is_MTK=is_MTK,is_simplify=is_simplify,is_bs=is_bs,
                    Ncons=Ncons,numMultistart=numMultistart,uhMax=uhMax,vhthMin=vhthMin,vhthMax=vhthMax,
                    maxIterKing=maxIterKing,rtol_OrjL=rtol_OrjL,show_trace=show_trace, 
                    x_tol=x_tol,f_tol=f_tol,g_tol=g_tol,
                    Nspan_optim_nuTi=Nspan_optim_nuTi)
    else
            xssr, is_converged, xfit, niter = optimMhjL!(naiLt0,uaiLt0,vthiLt0,uhLN,MhjL,L,nModL,DMh024;
                    NL_solve=NL_solve,Optlibary=Optlibary,NL_solve_method=NL_solve_method,optimizer=optimizer,
                    linsolve=linsolve,linesearch=linesearch,preconditioner=preconditioner,ADtype=ADtype, 
                    is_norm_uhL=is_norm_uhL,is_Jacobian=is_Jacobian,is_Hessian=is_Hessian,is_AD=is_AD,
                    is_constraint=is_constraint,is_MTK=is_MTK,is_simplify=is_simplify,is_bs=is_bs,
                    Ncons=Ncons,numMultistart=numMultistart,uhMax=uhMax,vhthMin=vhthMin,vhthMax=vhthMax,
                    maxIterKing=maxIterKing,rtol_OrjL=rtol_OrjL,show_trace=show_trace, 
                    x_tol=x_tol,f_tol=f_tol,g_tol=g_tol,
                    Nspan_optim_nuTi=Nspan_optim_nuTi)
    end
    @show L
    if is_norm_uhL
        @show fmtf2.([uhLN,uhLN^L])
    end
    if xssr < 1e-7
            printstyled("((is_re_seed, niter), xssr,linsolve)=",((is_re_seed, niter), fmt2(xssr),linsolve);color=:green)
    else
            printstyled("((is_re_seed, niter), xssr,linsolve)=",((is_re_seed, niter), fmt2(xssr),linsolve);color=:red)
    end
    # if is_converged
        naio = xfit[1:3:end]
        uaio = xfit[2:3:end]
    #     if uaiLt0[nModL] < 0.0
    #         uaio[nModL] *= - 1
    #     end
        vthio = xfit[3:3:end]
        MhjLo = zeros(datatype,njML)
        MhsKMM!(MhjLo,jvecL,L,naio,uaio,vthio,uhLN,nModL;is_renorm=is_renorm,
                is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL,mathtype=mathtype)
        
        RDn = Float64.(sum_kbn(naio) - 1.0)
        RDTT0 = (sum_kbn(naio .* vthio .^2) / sum_kbn(naiL .* vthiL .^2) - 1.0)
        RDTT = Float64.(RDTT0)
        RDEk = Float64.(sum_kbn(naio .* uaio .^2) / sum_kbn(naiL .* uaiL .^2) - 1.0)
        RDMs = Float64.(MhjLo[1:3nModL] ./ MhjL[1:3nModL] .- 1)
    # end
    println()
    @show Int.([is_C, is_norm_uhL, is_Jacobian]), RDnuT
    @show fmt2.(RDMs)
    # @show RDTT0
    if is_anasys_L
        if iseven(L)
            inLp = L / 2 + 1 |> Int
        else
            inLp = (L - 1) / 2 + 1 |> Int
        end
        errMhjL[:,inLp] = RDMs
        errnuTM[inLp,:]  = [RDn, RDTT, RDEk, norm(RDMs)]
    end
    if norm(RDMs) ≤ 1e-10
            printstyled("(RDn,RDTT,RDEk,RDMs)=",(fmt2(RDn),fmt2(RDTT),fmt2(RDEk),fmt2(norm(RDMs)));color=:blue)
    else
            printstyled("(RDn,RDTT,RDEk,RDMs)=",(fmt2(RDn),fmt2(RDTT),fmt2(RDEk),fmt2(norm(RDMs)));color=:red)
    end
    println()
    # @show fmt2.(naio ./ naiL .- 1)
    # @show fmt2.(abs.(uaio ./ uaiL) .- 1)
    # @show fmt2.(vthio ./ vthiL .- 1)  
    if is_show_Dc 
        MhjLoMatrix = DataFrame(reshape(Float64.(MhjLo),1,njML),:auto)
        nuToptMatrix = Float64.([naiL naio uaiL uaio vthiL vthio uaiL./vthiL uaio./vthio]) 
        nuToptName = ["naiL", "naio", "uaiL", "uaio", "vthiL", "vthio", "uhhL", "uhho"]
        nuTopt = DataFrame(nuToptMatrix,:auto)
        rename!(nuTopt,nuToptName)  
        @show MhjLoMatrix
        @show nuTopt
        println()
    end
end
