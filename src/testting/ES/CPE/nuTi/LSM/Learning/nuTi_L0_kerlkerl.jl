

if 1 == 1
    scales = 1.0 + RDnuT
    naiLt0,uaiLt0,vthiLt0 = scales * naiL0,scales * uaiL0,scales * vthiL0
    Mhr0s = naiLt0

    # Mhr2s = naiLt0 .* (vthiLt0.^2 + CjLk(2,1) * uaiLt0.^2)
    # # @show Mhj0[2] - sum_kbn(Mhr2s)
    # # @show sum_kbn(Mhr2s), Mhr2s

    # Or40 = zeros(nModL0)
    # Orj0Nb!(Or40,(uaiLt0./vthiLt0).^2,4,2,nModL0;rtol_OrjL=rtol_OrjL)
    # Mhr4s = (naiLt0 .* (vthiLt0).^4) .* (1 .+ Or40)
    # # @show Mhj0[3] - sum_kbn(Mhr4s)
    # uhh2 = 1.5 * (2.5 * ((Mhr2s ./ Mhr0s).^2 - (Mhr4s ./ Mhr0s))).^0.5

    # println("//////////////////////////////")
    # @show Mhr0s[nModL0],Mhr2s[nModL0],Mhr4s[nModL0] 
    if is_C
            xssr, is_converged, xfit, niter = optimMhjLC!(naiLt0,uaiLt0,vthiLt0,Mhj0,L,nModL0,NL_solve,DMh024;
                    rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh,
                    NL_solve_method=NL_solve_method,ADtype=ADtype,
                    is_Jacobian=is_Jacobian, is_Hessian=is_Hessian, is_constraint=is_constraint,
                    optimizer=optimizer,factor=factor,autodiff=autodiff,
                    show_trace=show_trace, maxIterKing=maxIterKing,
                    x_tol=x_tol,f_tol=f_tol,g_tol=g_tol,
                    Nspan_optim_nuTi=Nspan_optim_nuTi)
    else
            xssr, is_converged, xfit, niter = optimMhjL!(naiLt0,uaiLt0,vthiLt0,Mhj0,L,nModL0,NL_solve,DMh024;
                    rtol_OrjL=rtol_OrjL,
                    NL_solve_method=NL_solve_method,ADtype=ADtype,
                    is_Jacobian=is_Jacobian, is_Hessian=is_Hessian, is_constraint=is_constraint,
                    optimizer=optimizer,factor=factor,autodiff=autodiff,
                    show_trace=show_trace, maxIterKing=maxIterKing,
                    x_tol=x_tol,f_tol=f_tol,g_tol=g_tol,
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
        
        RDn = fmt2(sum_kbn(naio) - 1.0)
        RDTT = fmt2(sum_kbn(naio .* uaio .^2) / sum_kbn(naiL0 .* uaiL0 .^2) - 1.0)
        RDMs = (Mhj0o[1:3nModL0] ./ Mhj0[1:3nModL0] .- 1)
    # end
    println()
    @show is_C, is_re_seed, is_Jacobian, RDnuT
    @show fmt2.(RDMs)
    @show RDn,RDTT,norm(RDMs)
    # @show fmt2.(naio ./ naiL0 .- 1)
    # @show fmt2.(abs.(uaio ./ uaiL0) .- 1)
    # @show fmt2.(vthio ./ vthiL0 .- 1)  
    if is_show_Dc 
        nuToptMatric = [naiL0 naio uaiL0 uaio vthiL0 vthio uaiL0./vthiL0 uaio./vthio]  
        nuToptName = ["naiL0", "naio", "uaiL0", "uaio", "vthiL0", "vthio", "uhhL0", "uhho"]
        nuTopt = DataFrame(nuToptMatric,:auto)
        rename!(nuTopt,nuToptName)  
        @show nuTopt
        println()
    end
end
