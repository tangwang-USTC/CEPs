

if 1 == 1
    scales = 1.0 + RDnuT
    naiLt0,uaiLt0,vthiLt0 = scales * naiL,scales * uaiL,scales * vthiL
    Mhr0s = naiLt0

    # Mhr2s = naiLt0 .* (vthiLt0.^2 + CjLk(2,1) * uaiLt0.^2)
    # # @show MhjL[2] - sum(Mhr2s)
    # # @show sum(Mhr2s), Mhr2s

    # Or40 = zeros(nModL)
    # Orj0Nb!(Or40,(uaiLt0./vthiLt0).^2,4,2,nModL;rtol_OrjL=rtol_OrjL)
    # Mhr4s = (naiLt0 .* (vthiLt0).^4) .* (1 .+ Or40)
    # # @show MhjL[3] - sum(Mhr4s)
    # uhh2 = 1.5 * (2.5 * ((Mhr2s ./ Mhr0s).^2 - (Mhr4s ./ Mhr0s))).^0.5

    # println("//////////////////////////////")
    # @show Mhr0s[nModL],Mhr2s[nModL],Mhr4s[nModL] 
    if is_C
            xssr, is_converged, xfit, niter = optimMhjLC!(naiLt0,uaiLt0,vthiLt0,MhjL,L,nModL,NL_solve,DMh024;
                    rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh,
                    optimizer=optimizer,factor=factor,autodiff=autodiff,
                    is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                    p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,NL_solve_method=NL_solve_method,
                    Nspan_optim_nuTi=Nspan_optim_nuTi)
    else
            xssr, is_converged, xfit, niter = optimMhjL!(naiLt0,uaiLt0,vthiLt0,MhjL,L,nModL,NL_solve,DMh024;
                    rtol_OrjL=rtol_OrjL,
                    optimizer=optimizer,factor=factor,autodiff=autodiff,
                    is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                    p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,NL_solve_method=NL_solve_method,
                    Nspan_optim_nuTi=Nspan_optim_nuTi)
    end
    
    if is_converged
            printstyled("(L, niter, xssr,factor)=",(L, niter, fmt2(xssr),factor);color=:green)
    else
            printstyled("(L, niter, xssr,factor)=",(L, niter, fmt2(xssr),factor);color=:red)
    end
    # if is_converged
        naio = xfit[1:3:end]
        uaio = xfit[2:3:end]
    #     if uaiLt0[nModL] < 0.0
    #         uaio[nModL] *= - 1
    #     end
        vthio = xfit[3:3:end]
        MhjLo = zeros(njML)
        MhsKMM!(MhjLo,jvecL,L,naio,uaio,vthio,nModL;is_renorm=is_renorm,mathtype=mathtype)
        
        RDn = fmt2(sum(naio) - 1.0)
        RDTT = fmt2(sum(naio .* uaio .^2) / sum(naiL .* uaiL .^2) - 1.0)
        RDMs = (MhjLo[1:3nModL] ./ MhjL[1:3nModL] .- 1)
    # end
    println()
    @show is_C, is_re_seed, is_Jacobian, RDnuT
    @show fmt2.(RDMs)
    @show RDn,RDTT,norm(RDMs)
    # @show fmt2.(naio ./ naiL .- 1)
    # @show fmt2.(abs.(uaio ./ uaiL) .- 1)
    # @show fmt2.(vthio ./ vthiL .- 1)  
    if is_show_Dc 
        MhjLoMatrix = DataFrame(reshape(MhjLo,1,njML),:auto)
        nuToptMatrix = [naiL naio uaiL uaio vthiL vthio uaiL./vthiL uaio./vthio]  
        nuToptName = ["naiL", "naio", "uaiL", "uaio", "vthiL", "vthio", "uhhL", "uhho"]
        nuTopt = DataFrame(nuToptMatrix,:auto)
        rename!(nuTopt,nuToptName)  
        @show MhjLoMatrix
        @show nuTopt
        println()
    end
end
