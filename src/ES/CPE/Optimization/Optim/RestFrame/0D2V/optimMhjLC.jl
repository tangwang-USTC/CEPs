

"""
  Optimization `Nₖₐ`, characteristic parameters `(n̂ᵣ,ûᵣ,v̂ₜₕᵣ)` and 
    normalzied kinetic moments `𝓜ⱼ(f̂ₗ)`
    by solving the characteristic parameter equations (CPEs)
    for weakly anisotropic and moderate anisotropic plasma system 
    with general axisymmetric velocity space. 
    The plasma is in a local sub-equilibrium state.

    This version is based on the conservation-constrait CPEs (CPEsC)
    where CPEs with order `j ∈ {l,l+2,l+4}` are enforced by the algorithm.

  The general kinetic moments can be renormalized by `CMjL`.
    
    If `is_renorm == true`,
      `Mhst = 𝓜ⱼ(f̂ₗ) / CMjL`
    end
 
  Notes: `{M̂₁₁}/3 = Î = n̂ û`, generally.
  
  See Ref. of Wang (2025) titled as "StatNN: An inherently interpretable physics-informed neural networks designed for moderate anisotropic plasma simulation".
  
"""

"""
  Inputs:
    `Mhst = M̂ⱼₗ* := M̂ⱼₗ / CMⱼₗ ` where `j = L+0, L+2, L+4, ⋯, jₘₐₓ`
    `nai: = [n̂ᵣ]` where `r = 1, 2, ⋯, nMod`
  
  Outputs:
    optimMhjLC!(nai,uai,vthi,Mhst,L,nMod,NL_solve,DMh024;
                rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh,
                optimizer=optimizer,factor=factor,autodiff=autodiff,
                is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,NL_solve_method=NL_solve_method,
                Nspan_optim_nuTi=Nspan_optim_nuTi)
  
"""

# [nMod]
function optimMhjLC!(nai::AbstractVector{T}, uai::AbstractVector{T}, vthi::AbstractVector{T}, 
    Mhst::AbstractVector{T}, L::Int, nMod::Int, NL_solve::Symbol, DMh024::AbstractVector{T}; 
    rtol_OrjL::T=1e-10,atol_Mh::T=1e-10,rtol_Mh::T=1e-10,
    optimizer=Dogleg, factor=QR(), autodiff::Symbol=:central,
    is_Jacobian::Bool=true, show_trace::Bool=false, maxIterKing::Int=200,
    p_tol::Float64=epsT, f_tol::Float64=epsT, g_tol::Float64=epsT, 
    NL_solve_method::Symbol=:newton,Nspan_optim_nuTi::AbstractVector{T}=[1.1,1.1,1.1]) where {T}

    # vthi
    if nMod == 1
        nai[1], uai[1], vthi[1] = optimMhjLC(Mhst, L)
    else
        # The parameter limits for MCF plasma.
        nMod1 = nMod - 1
        x0 = zeros(3nMod1)      # [uai1, vthi1, uai2, vthi2, ⋯]
        lbs = zeros(3nMod1)
        ubs = zeros(3nMod1)

        vec = 1:nMod1
        for i in vec
            # nai
            i1 = 3i - 2
            lbs[i1] = 0
            ubs[i1] = 1.0
            x0[i1] = deepcopy(nai[i])

            # uai
            i2 = 3i - 1
            lbs[i2] = - uhMax
            ubs[i2] = uhMax
            x0[i2] = deepcopy(uai[i])

            # vthi
            i3 = 3i
            lbs[i3] = vhthMin
            ubs[i3] = vhthMax
            x0[i3] = deepcopy(vthi[i])
        end

        xssr, is_converged, xfit, niter = optimMhjLC!(x0,Mhst,L,nMod,lbs,ubs,NL_solve; 
            rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh,
            optimizer=optimizer, factor=factor, autodiff=autodiff,
            is_Jacobian=is_Jacobian, show_trace=show_trace, maxIterKing=maxIterKing,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, NL_solve_method=NL_solve_method)

        # if is_converged
        #     nai[1:nMod] = xfit[1:3:end]
        #     uai[1:nMod] = xfit[2:3:end]
        #     vthi[1:nMod] = xfit[3:3:end]
        # end
        return xssr, is_converged, xfit, niter
    end
end


"""
  Inputs:
    `Mhst = M̂ⱼₗ* := M̂ⱼₗ / CMⱼₗ ` where `j = L+0, L+2, L+4, ⋯, jₘₐₓ`
    `nai: = [n̂ᵣ]` where `r = 1, 2, ⋯, nMod`
  
  Outputs:
    is_converged_nMod = optimMhjLC!(x0,Mhst,nMod,lbs,ubs,NL_solve;
                rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh,
                optimizer=optimizer,factor=factor,autodiff=autodiff,
                is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,NL_solve_method=NL_solve_method)
  
"""

# nMode = 1
function optimMhjLC(Mhst::AbstractVector{T},L::Int) where{T}

    uh2 = ((L+2.5) * (Mhst[2]) - (L+1.5) * Mhst[3])^0.5
    if iseven(L)
        uh = (uh2)^0.5
    else
        uh = sign(Mhst[1]) * (uh2)^0.5
    end
    vhth = (1/(L+1.5) * (Mhst[2])^2 - uh2)^0.5
    nh = Mhst[1] / uh2 ^(L/2)
    return nh, uh, vhth
end

# [nMod ≥ 2]
function optimMhjLC!(x0::AbstractVector{T}, Mhst::AbstractVector{T}, L::Int, nMod::Int, 
    lbs::AbstractVector{T}, ubs::AbstractVector{T}, NL_solve::Symbol; 
    rtol_OrjL::T=1e-10,atol_Mh::T=1e-10,rtol_Mh::T=1e-10,
    optimizer=Dogleg, factor=QR(), autodiff::Symbol=:central,
    is_Jacobian::Bool=true, show_trace::Bool=false, maxIterKing::Int=200,
    p_tol::Float64=epsT, f_tol::Float64=epsT, g_tol::Float64=epsT,
    NL_solve_method::Symbol=:newton) where {T}

    nh,uh,vhth,vhth2,uvth2 = zeros(T,nMod),zeros(T,nMod),zeros(T,nMod),zeros(T,nMod),zeros(T,nMod),zeros(T,nMod)
    DMjL = zeros(T,3)
    J = zeros(3,3)
    DM1RjL = zeros(T,3)
    nMod1 = nMod - 1
    vth1jL = zeros(T,3nMod1) 
    vthijL = zeros(T,4)                # [vthiLL,vthi2L,vthi4L,vthijL]
    Vn1L = zeros(T,3)
    VnrL = zeros(T,3)
    M1jL = zeros(T,3)                  # := [M1LL, RM12L, RM14L]
    crjL = zeros(T,3)
    arLL = zeros(T,3)
    ar2L = zeros(T,3)
    ar4L = zeros(T,3)
    O1jL2 = zeros(T,3nMod1)
    O1jL = zeros(T,3nMod1)
    OrnL2 = zeros(T,3)                 # = [Or0L2, Or2L2, Or4L2]
    OrnL = zeros(T,3) 

    # res = optimMhjLC(x0, Mhst, L, nMod, NL_solve; 
    #     nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,DMjL=DMjL,J=J,
    #     DM1RjL=DM1RjL,Vn1L=Vn1L,VnrL=VnrL,M1jL=M1jL,vth1jL=vth1jL,
    #     O1jL2=O1jL2,O1jL=O1jL,OrnL2=OrnL2,OrnL=OrnL,
    #     lbs=lbs,ubs=ubs,optimizer=optimizer, factor=factor, autodiff=autodiff,
    #     is_Jacobian=is_Jacobian, show_trace=show_trace, maxIterKing=maxIterKing,
    #     p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, NL_solve_method=NL_solve_method)
    
    MhjL_GKMM!(out, x) = CPEjLC!(out, x, L, nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,M1jL=M1jL,
                                    rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
    if NL_solve == :LeastSquaresOptim
        if is_Jacobian
            J!(JM, x) = JacobCPEjLC!(JM, x, L, nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,
                        DMjL=DMjL,J=J,DM1RjL=DM1RjL,vth1jL=vth1jL,vthijL=vthijL,Vn1L=Vn1L,VnrL=VnrL,M1jL=M1jL,
                        crjL=crjL,arLL=arLL,ar2L=ar2L,ar4L=ar4L,O1jL2=O1jL2,O1jL=O1jL,
                        OrnL2=OrnL2,OrnL=OrnL,rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
            nls = LeastSquaresProblem(x=x0, (f!)=MhjL_GKMM!, (g!)=J!, output_length=length(x0), autodiff=autodiff)
        else
            nls = LeastSquaresProblem(x=x0, (f!)=MhjL_GKMM!, output_length=length(x0), autodiff=autodiff)
        end
        res = optimize!(nls, optimizer(factor), iterations=maxIterKing, show_trace=show_trace,
            x_tol=p_tol, f_tol=f_tol, g_tol=g_tol, lower=lbs, upper=ubs)
    elseif NL_solve == :NLsolve
        if is_Jacobian
            Js!(JM, x) = JacobCPEjLC!(JM, x, L, nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,
                        DMjL=DMjL,J=J,DM1RjL=DM1RjL,vth1jL=vth1jL,vthijL=vthijL,Vn1L=Vn1L,VnrL=VnrL,M1jL=M1jL,
                        crjL=crjL,arLL=arLL,ar2L=ar2L,ar4L=ar4L,O1jL2=O1jL2,O1jL=O1jL,
                        OrnL2=OrnL2,OrnL=OrnL,rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
            nls = OnceDifferentiable(MhjL_GKMM!, Js!, x0, similar(x0))
            if NL_solve_method == :trust_region
                res = nlsolve(nls, x0, method=NL_solve_method, factor=1.0, autoscale=true, xtol=p_tol, ftol=f_tol,
                    iterations=maxIterKing, show_trace=show_trace)
            elseif NL_solve_method == :newton
                res = nlsolve(nls, x0, method=NL_solve_method, xtol=p_tol, ftol=f_tol,
                    iterations=maxIterKing, show_trace=show_trace)
            end
        else
            nls = OnceDifferentiable(MhjL_GKMM!, x0, similar(x0))
            if NL_solve_method == :trust_region
                res = nlsolve(nls, x0, method=NL_solve_method, factor=1.0, autoscale=true, xtol=p_tol, ftol=f_tol,
                    iterations=maxIterKing, show_trace=show_trace, autodiff=:forward)
            elseif NL_solve_method == :newton
                res = nlsolve(nls, x0, method=NL_solve_method, xtol=p_tol, ftol=f_tol,
                    iterations=maxIterKing, show_trace=show_trace, autodiff=:forward)
            end
        end
    elseif NL_solve == :JuMP
        gyhhjjmj
    else
        esfgroifrtg
    end

    if NL_solve == :LeastSquaresOptim
        niter = res.iterations
        is_converged = res.converged
        xssr = res.ssr                         # sum(abs2, fcur)
        xfit = res.minimizer         # the vector of best model1 parameters
    elseif NL_solve == :NLsolve
        niter = res.iterations
        is_converged = res.f_converged
        xssr = res.residual_norm                         # sum(abs2, fcur)
        xfit = res.zero         # the vector of best model1 parameters
    elseif NL_solve == :JuMP
        fgfgg
    end
    # ccsj0!(nh,uh,vhth,vhth2,uvth2,Mhst[1:3],nMod)
    return xssr, is_converged, [xfit;nh[nMod];uh[nMod];vhth[nMod]], niter
end

"""
  Inputs:
    Mhst: = M̂ⱼₗ*, which is the renormalized general kinetic moments.
  
  Outputs:
  res = optimMhjLC(x0, Mhst, L, nMod, NL_solve; 
            nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,lbs=lbs,ubs=ubs,
            rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh,
            optimizer=optimizer, factor=factor, autodiff=autodiff,
            is_Jacobian=is_Jacobian, show_trace=show_trace, maxIterKing=maxIterKing,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, NL_solve_method=NL_solve_method)
  
"""

# [nMod ≥ 2]
function optimMhjLC(x0::AbstractVector{T}, Mhst::AbstractVector{T}, L::Int, nMod::Int, NL_solve::Symbol;
    nh::AbstractVector{T}=[0.1, 1.0], uh::AbstractVector{T}=[0.1, 1.0], vhth::AbstractVector{T}=[0.1, 1.0], 
    vhth2::AbstractVector{T}=[0.1, 1.0], uvth2::AbstractVector{T}=[0.1, 1.0],DMjL::AbstractVector{T}=[0.0,0.0,0.0],
    J::AbstractArray{T,N2}=[0.0 0.0;0.0 0.0],DM1RjL::AbstractVector{T}=[0.0,0.0,0.0],
    Vn1L::AbstractVector{T}=[0.0,0.0,0.0],VnrL::AbstractVector{T}=[0.0,0.0,0.0],
    M1jL::AbstractVector{T}=[1.0,1.0,1.0],vth1jL::AbstractVector{T}=[0.0,0.0,0.0],
    O1jL2::AbstractVector{T}=[0.0,0.0,0.0],O1jL::AbstractVector{T}=[0.0,0.0,0.0],
    OrnL2::AbstractVector{T}=[0.0,0.0,0.0],OrnL::AbstractVector{T}=[0.0,0.0,0.0], 
    lbs::AbstractVector{T}=[-uhMax, 0.8], ubs::AbstractVector{T}=[uhMax, 1.2], 
    rtol_OrjL::T=1e-10,atol_Mh::T=1e-10,rtol_Mh::T=1e-10,
    optimizer=Dogleg, factor=QR(), autodiff::Symbol=:central,
    is_Jacobian::Bool=true, show_trace::Bool=false, maxIterKing::Int=200,
    p_tol::Float64=epsT, f_tol::Float64=epsT, g_tol::Float64=epsT,
    NL_solve_method::Symbol=:newton) where {T,N2}

    MhjL_GKMM!(out, x) = CPEjLC!(out, x, L, nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,M1jL=M1jL,
                                    rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
    if NL_solve == :LeastSquaresOptim
        if is_Jacobian
            J!(JM, x) = JacobCPEjLC!(JM, x, L, nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,
                        DMjL=DMjL,J=J,DM1RjL=DM1RjL,vth1jL=vth1jL,vthijL=vthijL,Vn1L=Vn1L,VnrL=VnrL,M1jL=M1jL,
                        crjL=crjL,arLL=arLL,ar2L=ar2L,ar4L=ar4L,O1jL2=O1jL2,O1jL=O1jL,
                        OrnL2=OrnL2,OrnL=OrnL,rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
            nls = LeastSquaresProblem(x=x0, (f!)=MhjL_GKMM!, (g!)=J!, output_length=length(x0), autodiff=autodiff)
        else
            nls = LeastSquaresProblem(x=x0, (f!)=MhjL_GKMM!, output_length=length(x0), autodiff=autodiff)
        end
        res = optimize!(nls, optimizer(factor), iterations=maxIterKing, show_trace=show_trace,
            x_tol=p_tol, f_tol=f_tol, g_tol=g_tol, lower=lbs, upper=ubs)
    elseif NL_solve == :NLsolve
        if is_Jacobian
            Js!(JM, x) = JacobCPEjLC!(JM, x, L, nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,
                        DMjL=DMjL,J=J,DM1RjL=DM1RjL,vth1jL=vth1jL,vthijL=vthijL,Vn1L=Vn1L,VnrL=VnrL,M1jL=M1jL,
                        crjL=crjL,arLL=arLL,ar2L=ar2L,ar4L=ar4L,O1jL2=O1jL2,O1jL=O1jL,
                        OrnL2=OrnL2,OrnL=OrnL,rtol_OrjL=rtol_OrjL,atol_Mh=atol_Mh,rtol_Mh=rtol_Mh)
            nls = OnceDifferentiable(MhjL_GKMM!, Js!, x0, similar(x0))
            if NL_solve_method == :trust_region
                res = nlsolve(nls, x0, method=NL_solve_method, factor=1.0, autoscale=true, xtol=p_tol, ftol=f_tol,
                    iterations=maxIterKing, show_trace=show_trace)
            elseif NL_solve_method == :newton
                res = nlsolve(nls, x0, method=NL_solve_method, xtol=p_tol, ftol=f_tol,
                    iterations=maxIterKing, show_trace=show_trace)
            end
        else
            nls = OnceDifferentiable(MhjL_GKMM!, x0, similar(x0))
            if NL_solve_method == :trust_region
                res = nlsolve(nls, x0, method=NL_solve_method, factor=1.0, autoscale=true, xtol=p_tol, ftol=f_tol,
                    iterations=maxIterKing, show_trace=show_trace, autodiff=:forward)
            elseif NL_solve_method == :newton
                res = nlsolve(nls, x0, method=NL_solve_method, xtol=p_tol, ftol=f_tol,
                    iterations=maxIterKing, show_trace=show_trace, autodiff=:forward)
            end
        end
    elseif NL_solve == :JuMP
        gyhhjjmj
    else
        esfgroifrtg
    end
    return res
end
