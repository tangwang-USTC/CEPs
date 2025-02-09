

"""
  Optimization `Nₖₐ`, characteristic parameters `(n̂ᵣ,ûᵣ,v̂ₜₕᵣ)` and 
    normalzied kinetic moments `𝓜ⱼ(f̂ₗ)`
    by solving the characteristic parameter equations (CPEs) 
    for weakly anisotropic and moderate anisotropic plasma system 
    with general axisymmetric velocity space. 
    The plasma is in a local sub-equilibrium state.

  The general kinetic moments are renormalized by `CMjL`.
    
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
    is_converged_nMod = optimMhjL!(nai,uai,vthi,uhLN,Mhst,L,nMod,DMh024;NL_solve=NL_solve,
                Optlibary=Optlibary,NL_solve_method=NL_solve_method,optimizer=optimizer,
                linsolve=linsolve,linesearch=linesearch,preconditioner=preconditioner,ADtype=ADtype, 
                is_norm_uhL=is_norm_uhL,is_Jacobian=is_Jacobian,is_Hessian=is_Hessian,is_AD=is_AD,
                is_constraint=is_constraint,is_MTK=is_MTK,is_simplify=is_simplify,is_bs=is_bs,
                Ncons=Ncons,numMultistart=numMultistart,uhMax=uhMax,vhthMin=vhthMin,vhthMax=vhthMax,
                maxIterKing=maxIterKing,rtol_OrjL=rtol_OrjL,show_trace=show_trace, 
                x_tol=x_tol,f_tol=f_tol,g_tol=g_tol,
                Nspan_optim_nuTi=Nspan_optim_nuTi)
  
""" 

# [nMod]
function optimMhjL!(nai::AbstractVector{T}, uai::AbstractVector{T}, vthi::AbstractVector{T}, uhLN::T, 
    Mhst::AbstractVector{T}, L::Int, nMod::Int, DMh024::AbstractVector{T}; NL_solve::Symbol=:Optimization, 
    Optlibary::Symbol=:Optimization, NL_solve_method::Symbol=:newton, optimizer=BFGS(), 
    linsolve=:qr, linesearch=nothing, preconditioner=nothing, ADtype=AutoEnzyme(),
    is_norm_uhL::Bool=true, is_Jacobian::Bool=true, is_Hessian::Bool=true, is_AD::Bool=true, 
    is_constraint::Bool=false, is_MTK::Bool=true, is_simplify::Bool=true, is_bs::Bool=true, 
    Ncons::Int=1, numMultistart::Int=1, uhMax::T=-3.0, vhthMin::T=1e-2, vhthMax::T=100, 
    maxIterKing::Int=200, rtol_OrjL::T=1e-10, show_trace::Bool=false, 
    x_tol::Float64=epsT, f_tol::Float64=epsT, g_tol::Float64=epsT,
    Nspan_optim_nuTi::AbstractVector{T}=[1.1,1.1,1.1]) where {T}


    if nMod == 1
        nai[1], uai[1], vthi[1] = optimMhjL(Mhst, L)
    else
        lbs = zeros(T,3nMod)
        ubs = zeros(T,3nMod)
        if NL_solve == :Optimization
            if is_MTK 
                xx = averiables_xvec(3nMod) 
                x0 = Vector{Pair{Num, T}}(undef,3nMod)
                xccCPEs!(x0,lbs,ubs,xx,nai,uai,vthi,nMod;uhMax=uhMax,vhthMin=vhthMin,vhthMax=vhthMax)
                xssr, is_converged, xfit, niter = optimMhjL!(xx,x0,uhLN,Mhst,L,nMod;lbs=lbs,ubs=ubs, 
                    NL_solve=NL_solve,Optlibary=Optlibary,NL_solve_method=NL_solve_method,optimizer=optimizer,
                    linsolve=linsolve,linesearch=linesearch,preconditioner=preconditioner,ADtype=ADtype, 
                    is_norm_uhL=is_norm_uhL,is_Jacobian=is_Jacobian,is_Hessian=is_Hessian,is_AD=is_AD,
                    is_constraint=is_constraint,is_simplify=is_simplify,is_bs=is_bs,
                    Ncons=Ncons,numMultistart=numMultistart,
                    maxIterKing=maxIterKing,rtol_OrjL=rtol_OrjL,show_trace=show_trace, 
                    x_tol=x_tol,f_tol=f_tol,g_tol=g_tol)
            else
                x0 = zeros(T,3nMod)      # [uai1, vthi1, uai2, vthi2, ⋯]
                xccCPEs!(x0,lbs,ubs,nai,uai,vthi,nMod;uhMax=uhMax,vhthMin=vhthMin,vhthMax=vhthMax)
                xssr, is_converged, xfit, niter = optimMhjL!(x0,uhLN,Mhst,L,nMod;lbs=lbs,ubs=ubs, 
                    NL_solve=NL_solve,Optlibary=Optlibary,NL_solve_method=NL_solve_method,optimizer=optimizer,
                    linsolve=linsolve,linesearch=linesearch,preconditioner=preconditioner,ADtype=ADtype, 
                    is_norm_uhL=is_norm_uhL,is_Jacobian=is_Jacobian,is_Hessian=is_Hessian,is_AD=is_AD,
                    is_constraint=is_constraint,is_MTK=false,is_bs=is_bs,
                    Ncons=Ncons,numMultistart=numMultistart,
                    maxIterKing=maxIterKing,rtol_OrjL=rtol_OrjL,show_trace=show_trace, 
                    x_tol=x_tol,f_tol=f_tol,g_tol=g_tol)
            end
        elseif NL_solve == :NonLinearSolve
        else
            x0 = zeros(T,3nMod)      # [uai1, vthi1, uai2, vthi2, ⋯]
            xccCPEs!(x0,lbs,ubs,nai,uai,vthi,nMod;uhMax=uhMax,vhthMin=vhthMin,vhthMax=vhthMax)
            xssr, is_converged, xfit, niter = optimMhjL!(x0,uhLN,Mhst,L,nMod;lbs=lbs,ubs=ubs, 
                NL_solve=NL_solve,Optlibary=Optlibary,NL_solve_method=NL_solve_method,optimizer=optimizer,
                linsolve=linsolve,linesearch=linesearch,preconditioner=preconditioner,ADtype=ADtype, 
                is_norm_uhL=is_norm_uhL,is_Jacobian=is_Jacobian,is_Hessian=is_Hessian,is_AD=is_AD,
                is_constraint=is_constraint,is_MTK=is_MTK,is_bs=is_bs,numMultistart=numMultistart,
                maxIterKing=maxIterKing,rtol_OrjL=rtol_OrjL,show_trace=show_trace, 
                x_tol=x_tol,f_tol=f_tol,g_tol=g_tol)
        end


        # if is_converged
        #     vec = 1:nMod
        #     nai[vec] = xfit[1:3:end]
        #     uai[vec] = xfit[2:3:end]
        #     vthi[vec] = xfit[3:3:end]
        # end
        return xssr, is_converged, xfit, niter
    

        # uai[nai .≤ atol_n] .= 0.0
        # uhafit = sum_kbn(nai .* uai)
    
        # DMh024[1] = sum_kbn(nai) - 1
    
        # # # T̂ = ∑ₖ n̂ₖ (v̂ₜₕₖ² + 2/3 * ûₖ²) - 2/3 * û², where `û = ∑ₖ(n̂ₖûₖ) / ∑ₖ(n̂ₖ)`
        # # Thfit = sum_kbn(nai .* (vthi .^ 2 + 2 / 3 * uai .^ 2)) - 2 / 3 * uhafit .^ 2
        # DMh024[3] = sum_kbn(nai .* (vthi .^ 2 + 2 / 3 * uai .^ 2)) - 2 / 3 * uhafit .^ 2 -1      # Thfit .- 1
    end
end

"""
  Inputs:
    `Mhst = M̂ⱼₗ* := M̂ⱼₗ / CMⱼₗ ` where `j = L+0, L+2, L+4, ⋯, jₘₐₓ`
    `nai: = [n̂ᵣ]` where `r = 1, 2, ⋯, nMod`
  
  Outputs:
    is_converged_nMod = optimMhjL!(x0,uhLN,Mhst,L,nMod;lbs=lbs,ubs=ubs,NL_solve=NL_solve,
                Optlibary=Optlibary,NL_solve_method=NL_solve_method,optimizer=optimizer,
                linsolve=linsolve,linesearch=linesearch,preconditioner=preconditioner,ADtype=ADtype, 
                is_norm_uhL=is_norm_uhL,is_Jacobian=is_Jacobian,is_Hessian=is_Hessian,is_AD=is_AD,
                is_constraint=is_constraint,is_MTK=is_MTK,is_bs=is_bs,
                Ncons=Ncons,numMultistart=numMultistart,
                maxIterKing=maxIterKing,rtol_OrjL=rtol_OrjL,show_trace=show_trace, 
                x_tol=x_tol,f_tol=f_tol,g_tol=g_tol)
    is_converged_nMod = optimMhjL!(xx,x0,uhLN,Mhst,L,nMod;lbs=lbs,ubs=ubs,NL_solve=NL_solve,
                Optlibary=Optlibary,NL_solve_method=NL_solve_method,optimizer=optimizer,
                linsolve=linsolve,linesearch=linesearch,preconditioner=preconditioner,ADtype=ADtype, 
                is_norm_uhL=is_norm_uhL,is_Jacobian=is_Jacobian,is_Hessian=is_Hessian,is_AD=is_AD,
                is_constraint=is_constraint,is_simplify=is_simplify,is_bs=is_bs,
                Ncons=Ncons,numMultistart=numMultistart,
                maxIterKing=maxIterKing,rtol_OrjL=rtol_OrjL,show_trace=show_trace, 
                x_tol=x_tol,f_tol=f_tol,g_tol=g_tol)
  
"""

# [nMod] 
function optimMhjL!(x0::AbstractVector{T}, uhLN::T, Mhst::AbstractVector{T}, L::Int, nMod::Int;
    lbs::AbstractVector{T}=[0.1], ubs::AbstractVector{T}=[0.1], NL_solve::Symbol=:Optimization, 
    Optlibary::Symbol=:Optimization, NL_solve_method::Symbol=:newton, optimizer=BFGS, 
    linsolve=:qr, linesearch=nothing, preconditioner=nothing, ADtype=AutoEnzyme(),
    is_norm_uhL::Bool=true, is_Jacobian::Bool=true, is_Hessian::Bool=true, is_AD::Bool=true,
    is_constraint::Bool=false, is_MTK::Bool=true, is_bs::Bool=false, 
    Ncons::Int=1, numMultistart::Int=1, maxIterKing::Int=200, rtol_OrjL::T=1e-10, show_trace::Bool=false, 
    x_tol::Float64=epsT, f_tol::Float64=epsT, g_tol::Float64=epsT) where {T <: Real}

    nh,uh,vhth,vhth2,uvth2 = zeros(T,nMod),zeros(T,nMod),zeros(T,nMod),zeros(T,nMod),zeros(T,nMod),zeros(T,nMod)
    
    if NL_solve == :Optimization
        if is_MTK
            @variables nh[1:3nMod] [bounds = (0.0, 1.0)]
            @variables uh[1:3nMod] [bounds = (- uhMax, uhMax)]
            @variables Th[1:3nMod] [bounds = (vhthMin, vhthMax)]
            cons = consMhjLeqs(nh,uh,Th)
            @named optsys = OptimizationSystem()
        else
            # Defining the special form for specified solvor with Pkg X.
            if is_constraint
                Ncons ≥ 1 || throw(ArgumentError("Please specify `1 ≤ Ncons < 3nMod` here."))
                p = [Ncons; Mhst]
                # consMhjL!(res,xx,p)
                lcons = zeros(T,Ncons)
                ucons = zeros(T,Ncons) 

                MhjLc_GKMM(x,p=nothing) = CPEjL(x,uhLN,L,nMod;Mhst=Mhst,is_norm_uhL=is_norm_uhL)
                # MhjLc_GKMM(x,p=nothing) = CPEjL(x,uhLN,L,nMod,Ncons;Mhst=Mhst,is_norm_uhL=is_norm_uhL)

                # Defining the objective function `f(x,p)`
                optf = Optimization.OptimizationFunction(MhjLc_GKMM,ADtype;cons=consMhjL!)
                if is_Jacobian 
                    if is_AD
                    else
                        gOc!(gM,x0,p=nothing) = gacobCPEjL!(gM,x0,uhLN,L,nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,
                                                is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL)
                    end
                    if numMultistart == 1
                        if is_bs 
                            nls = Optimization.OptimizationProblem(optf,x0,p;lb=lbs,ub=ubs,lcons=lcons,ucons=ucons)
                        else
                            nls = Optimization.OptimizationProblem(optf,x0,p;lcons=lcons,ucons=ucons)
                        end
                    else
                        if is_bs 
                            nls = Optimization.OptimizationProblem(optf,x0,p;lb=lbs,ub=ubs,lcons=lcons,ucons=ucons)
                        else
                            nls = Optimization.OptimizationProblem(optf,x0,p;lcons=lcons,ucons=ucons)
                        end
                    end
                else
                    if numMultistart == 1
                        if is_bs 
                            nls = Optimization.OptimizationProblem(optf,x0,p;lb=lbs,ub=ubs,lcons=lcons,ucons=ucons)
                        else
                            nls = Optimization.OptimizationProblem(optf,x0,p;lcons=lcons,ucons=ucons)
                        end
                    else
                        if is_bs 
                            nls = Optimization.OptimizationProblem(optf,x0,p;lb=lbs,ub=ubs,lcons=lcons,ucons=ucons)
                        else
                            nls = Optimization.OptimizationProblem(optf,x0,p;lcons=lcons,ucons=ucons)
                        end
                    end
                end
                res = solve(nls, optimizer();maxiters=maxIterKing)
            else
                MhjL_GKMM(x,p=nothing) = CPEjL(x,uhLN,L,nMod;Mhst=Mhst,is_norm_uhL=is_norm_uhL)
                if is_Jacobian 
                    if is_AD 
                        optf = Optimization.OptimizationFunction(MhjL_GKMM,ADtype)
                    else
                        gO!(gM,x0,p=nothing) = gacobCPEjL!(gM,x0,uhLN,L,nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,
                                                is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL)
                        optf = Optimization.OptimizationFunction(MhjL_GKMM;grad=gO!)
                    end
                    if is_bs 
                        nls = Optimization.OptimizationProblem(optf,x0;lb=lbs,ub=ubs)
                    else
                        nls = Optimization.OptimizationProblem(optf,x0)
                    end
                    if numMultistart == 1
                    else
                    end
                else
                    optf = Optimization.OptimizationFunction(MhjL_GKMM,ADtype)
                    if numMultistart == 1
                        if is_bs 
                            nls = Optimization.OptimizationProblem(optf,x0;lb=lbs,ub=ubs)
                        else
                            nls = Optimization.OptimizationProblem(optf,x0)
                        end
                        res = solve(nls, optimizer();maxiters=maxIterKing)
                    else
                        if is_bs 
                            nls = Optimization.OptimizationProblem(optf,x0;lb=lbs,ub=ubs)
                        else
                            nls = Optimization.OptimizationProblem(optf,x0)
                        end 
                        res = solve(nls, optimizer();maxiters=maxIterKing)
                        ygygggg
                    end
                end
            end
        end
    elseif NL_solve == :NonlinearSolve
        # Defining the objective function `f(x,p)`
        x = averiables_xvec(3nMod) 
        out = Vector{Equation}(undef,3nMod)
        # nls = NonlinearLeastSquaresProblem(fnls!, u0)
        if is_Jacobian
            if is_AD
            else
            end
            Js21!(JM, x) = JacobCPEjL!(JM,x,uhLN,L,nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,
                                    is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL)
            nls = NonlinearLeastSquaresProblem(NonlinearFunction(MhjL_GKMM!,jac==Js21!,resid_prototype = zeros(length(x0))), x0)      # , p
        else
            nls = NonlinearLeastSquaresProblem(NonlinearFunction(MhjL_GKMM!,resid_prototype = zeros(length(x0))), x0)
        end
        
        if Optlibary == :LeastSquaresOptim
            res = NonlinearSolve.solve(nls, LeastSquaresOptimJL(optimizer;linsolve=linsolve,autodiff=ADtype);maxiters=maxIterKing)
        elseif Optlibary == :NonlinearSolve  
        else
        end
    else
        MhjL_GKMM!(out, x) = CPEjL!(out,x,uhLN,L,nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,
                                    is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL)
        
        if NL_solve == :LeastSquaresOptim
            if is_Jacobian
                if is_AD
                else
                end
                Js1!(JM, x) = JacobCPEjL!(JM,x,uhLN,L,nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,
                                        is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL)
                nls = LeastSquaresProblem(x=x0, (f!)=MhjL_GKMM!, (g!)=Js1!, output_length=length(x0), autodiff=ADtype)
            else
                nls = LeastSquaresProblem(x=x0, (f!)=MhjL_GKMM!, output_length=length(x0), autodiff=ADtype)
            end
            res = optimize!(nls, optimizer(linsolve), iterations=maxIterKing, show_trace=show_trace,
                x_tol=x_tol, f_tol=f_tol, g_tol=g_tol, lower=lbs, upper=ubs)
        elseif NL_solve == :NLsolve
            if is_Jacobian
                if is_AD
                else
                end
                Js!(JM, x) = JacobCPEjL!(JM,x,uhLN,L,nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,
                                        is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL)
                nls = OnceDifferentiable(MhjL_GKMM!, Js!, x0, similar(x0))
                if NL_solve_method == :trust_region
                    res = nlsolve(nls, x0, method=optimizer, factor=1.0, autoscale=true, xtol=x_tol, ftol=f_tol,
                        iterations=maxIterKing, show_trace=show_trace)
                elseif NL_solve_method == :newton
                    res = nlsolve(nls, x0, method=optimizer, xtol=x_tol, ftol=f_tol,
                        iterations=maxIterKing, show_trace=show_trace)
                end
            else
                nls = OnceDifferentiable(MhjL_GKMM!, x0, similar(x0))
                if NL_solve_method == :trust_region
                    res = nlsolve(nls, x0, method=optimizer, factor=1.0, autoscale=true, xtol=x_tol, ftol=f_tol,
                        iterations=maxIterKing, show_trace=show_trace, autodiff=ADtype)
                elseif NL_solve_method == :newton
                    res = nlsolve(nls, x0, method=optimizer, xtol=x_tol, ftol=f_tol,
                        iterations=maxIterKing, show_trace=show_trace, autodiff=ADtype)
                end
            end
        else
            esfgroifrtg
        end
    end
    # @show res
    # rrrrr

    if NL_solve == :Optimization
        xfit = res.u         # the vector of best model1 parameters
        niter = res.stats.iterations
        is_converged = res.retcode
        xssr = res.objective                         # sum_kbn(abs2, fcur)
        # alg = res.alg
        # caches = res.cache
        # time = res.stats.time
        # fevals = res.stats.fevals
        # gevals = res.stats.gevals
        # hevals = res.stats.hevals
        # original = res.original
    elseif NL_solve == :NonlinearSolve
    else
        if NL_solve == :LeastSquaresOptim
            xfit = res.minimizer         # the vector of best model1 parameters
            niter = res.iterations
            is_converged = res.converged
            xssr = res.ssr                         # sum_kbn(abs2, fcur)
        elseif NL_solve == :NLsolve
            xfit = res.zero         # the vector of best model1 parameters
            niter = res.iterations
            is_converged = res.f_converged
            xssr = res.residual_norm                         # sum_kbn(abs2, fcur)
        elseif NL_solve == :JuMP
            fgfgg
        end
    end
    return xssr, is_converged, xfit, niter
end

# `is_MTK=true`
function optimMhjL!(xx::AbstractVector{Num}, x0::AbstractVector{Pair{Num,T}}, uhLN::T, Mhst::AbstractVector{T}, L::Int, nMod::Int;
    lbs::AbstractVector{T}=[0.1], ubs::AbstractVector{T}=[0.1], NL_solve::Symbol=:Optimization, 
    Optlibary::Symbol=:Optimization, NL_solve_method::Symbol=:newton, optimizer=BFGS, 
    linsolve=:qr, linesearch=nothing, preconditioner=nothing, ADtype=AutoEnzyme(),
    is_norm_uhL::Bool=true, is_Jacobian::Bool=true, is_Hessian::Bool=true, is_AD::Bool=true,
    is_constraint::Bool=false, is_simplify::Bool=true, is_bs::Bool=false, 
    Ncons::Int=1, numMultistart::Int=1, maxIterKing::Int=200, rtol_OrjL::T=1e-10, show_trace::Bool=false, 
    x_tol::Float64=epsT, f_tol::Float64=epsT, g_tol::Float64=epsT) where {T}

    nh,uh,vhth,vhth2,uvth2 = zeros(T,nMod),zeros(T,nMod),zeros(T,nMod),zeros(T,nMod),zeros(T,nMod),zeros(T,nMod)
    
    nMod ≥ 2 || throw(ArgumentError("`nMod` must be no less than 2 here."))
    if NL_solve == :Optimization
        # Defining the special form for specified solvor with Pkg X.
        if is_constraint
            # Ncons ≥ 1 || throw(ArgumentError("Please specify `1 ≤ Ncons < 3nMod` here."))
            # Defining the objective function `f(x,p)`
            if Ncons == 0
                cons = [ ]
                out = zeros(Num,3nMod)
                CPEjL!(out,xx,uhLN,L,nMod,Ncons;Mhst=Mhst,is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL)
            else
                # if Ncons == 1
                #     cons = [0.0 ~ sum(xx[1:3:end] .* xx[2:3:end].^L) - Mhst[1]]
                # elseif Ncons == 2
                #     cons = [0.0 ~ sum(xx[1:3:end] .* xx[2:3:end].^L) - Mhst[1]
                #             0.0 ~ sum((xx[1:3:end] .* (xx[3:3:end]).^2 .* xx[2:3:end].^L) .* (1 .+ CjLk(2,L,1) * (xx[2:3:end]).^2 ./ (xx[3:3:end]).^2)) - Mhst[2]]
                # elseif Ncons == 3
                #     cons = [0.0 ~ sum(xx[1:3:end] .* xx[2:3:end].^L) - Mhst[1]
                #             0.0 ~ sum((xx[1:3:end] .* (xx[3:3:end]).^2 .* xx[2:3:end].^L) .* (1 .+ CjLk(2,L,1) * (xx[2:3:end]).^2 ./ (xx[3:3:end]).^2)) - Mhst[2]
                #             0.0 ~ sum((xx[1:3:end] .* (xx[3:3:end]).^2 .* xx[2:3:end].^L) .* (1 .+ CjLk(4,L,1) * (xx[2:3:end]).^2 ./ (xx[3:3:end]).^2 .+ CjLk(4,L,2) * (xx[2:3:end]).^4 ./ (xx[3:3:end]).^4)) - Mhst[3]]
                # else
                #     deedfggg
                # end
                # @variables xx[1:3nMod]
                if Ncons == 1
                    cons = [Mhst[1] ~ sum(xx[1:3:end] .* xx[2:3:end].^L)]
                elseif Ncons == 2
                    cons = [Mhst[1] ~ sum(xx[1:3:end] .* xx[2:3:end].^L)
                            Mhst[2] ~ sum((xx[1:3:end] .* (xx[3:3:end]).^2 .* xx[2:3:end].^L) .* (1 .+ CjLk(2,L,1) * (xx[2:3:end]).^2 ./ (xx[3:3:end]).^2))]
                elseif Ncons == 3
                    cons = [Mhst[1] ~ sum(xx[1:3:end] .* xx[2:3:end].^L)
                            Mhst[2] ~ sum((xx[1:3:end] .* (xx[3:3:end]).^2 .* xx[2:3:end].^L) .* (1 .+ CjLk(2,L,1) * (xx[2:3:end]).^2 ./ (xx[3:3:end]).^2))
                            Mhst[3] ~ sum((xx[1:3:end] .* (xx[3:3:end]).^2 .* xx[2:3:end].^L) .* (1 .+ CjLk(4,L,1) * (xx[2:3:end]).^2 ./ (xx[3:3:end]).^2 .+ CjLk(4,L,2) * (xx[2:3:end]).^4 ./ (xx[3:3:end]).^4))]
                else
                    deedfggg
                end
                out = zeros(Num,(3nMod - Ncons))
                CPEjL!(out,xx,uhLN,L,nMod,Ncons;Mhst=Mhst[(Ncons+1):end],is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL)
                @show xx
                @show cons
                # @show out
            end
            @named optf = OptimizationSystem(sum(abs2,out),xx,[];constraints=cons)
            # @named optf = OptimizationSystem(sum(abs2,out),xx,[])
            @show sum(abs2,out)
            # optf = complete(optf)
            if is_simplify
                optf = structural_simplify(optf)
                @show observed(optf)
            end
            if numMultistart == 1
                if is_Jacobian 
                    if is_AD
                    else
                        # gOMTKc!(gM,xx,p=nothing) = gacobCPEjL!(gM,xx,uhLN,L,nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,
                        #                         is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL)
                    end
                    # # Defining the special form for specified solvor with Pkg X.
                    if is_bs 
                        nls = Optimization.OptimizationProblem(optf,x0;lb=lbs,ub=ubs,grad=true,hess=true,cons_j=true,cons_h=true)
                    else
                        nls = Optimization.OptimizationProblem(optf,x0;grad=true,hess=true,cons_j=true,cons_h=true)
                    end
                else
                    if is_bs 
                        nls = Optimization.OptimizationProblem(optf,x0;grad=true,hess=true,cons_j=true,cons_h=true,lb=lbs,ub=ubs)
                    else
                        nls = Optimization.OptimizationProblem(optf,x0;grad=true,hess=true,cons_j=true,cons_h=true)
                    end
                end
            else
                if is_Jacobian 
                    if is_AD
                    else
                        # gOMTKcn!(gM,xx,p=nothing) = gacobCPEjL!(gM,xx,uhLN,L,nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,
                        #                         is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL)
                    end
                    # # Defining the special form for specified solvor with Pkg X.
                    if is_bs 
                        nls = Optimization.OptimizationProblem(optf,x0;lb=lbs,ub=ubs,grad=is_Jacobian,hess=is_Hessian,cons_j=is_Jacobian,cons_h=is_Hessian)
                    else
                        nls = Optimization.OptimizationProblem(optf,x0;grad=is_Jacobian,hess=is_Hessian,cons_j=is_Jacobian,cons_h=is_Hessian)
                    end
                else
                    if is_bs 
                        nls = Optimization.OptimizationProblem(optf,x0;grad=true,hess=true,cons_j=true,cons_h=true,lb=lbs,ub=ubs)
                    else
                        nls = Optimization.OptimizationProblem(optf,x0;grad=true,hess=true,cons_j=true,cons_h=true)
                    end
                end
            end
        else
            # Defining the objective function `f(x,p)`
            out = zeros(Num,3nMod)
            CPEjL!(out,xx,uhLN,L,nMod;Mhst=Mhst,is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL)
            @named optf = OptimizationSystem(sum(abs2,out),xx,[])
            # optf = complete(optf)
            if is_simplify
                optf = structural_simplify(optf)
                @show observed(optf)
            end
            if numMultistart == 1
                if is_Jacobian 
                    if is_AD
                    else
                        gOMTK!(gM,xx,p=nothing) = gacobCPEjL!(gM,xx,uhLN,L,nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,
                                                is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL)
                    end
                    if is_bs 
                        nls = Optimization.OptimizationProblem(optf,x0;lb=lbs,ub=ubs,grad=is_Jacobian,hess=is_Hessian)
                    else
                        nls = Optimization.OptimizationProblem(optf,x0;grad=is_Jacobian,hess=is_Hessian)
                    end
                else
                    if is_bs 
                        nls = Optimization.OptimizationProblem(optf,x0;lb=lbs,ub=ubs,grad=true,hess=true)
                    else
                        nls = Optimization.OptimizationProblem(optf,x0;grad=true,hess=true)
                        # nls = Optimization.OptimizationProblem(optf,x0;grad=true)
                    end
                end
            else
                if is_Jacobian 
                    if is_AD
                    else
                        gOMTKn!(gM,xx,p=nothing) = gacobCPEjL!(gM,xx,uhLN,L,nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,
                                                is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL)
                    end
                    if is_bs 
                        nls = Optimization.OptimizationProblem(optf,x0;lb=lbs,ub=ubs,grad=is_Jacobian,hess=is_Hessian)
                    else
                        nls = Optimization.OptimizationProblem(optf,x0;grad=is_Jacobian,hess=is_Hessian)
                    end
                else
                    if is_bs 
                        nls = Optimization.OptimizationProblem(optf,x0;lb=lbs,ub=ubs)
                    else
                        nls = Optimization.OptimizationProblem(optf,x0)
                    end
                end
            end
        end
        @show is_norm_uhL,is_Jacobian,is_Hessian,is_AD
        @show (is_constraint,is_simplify,is_bs),Ncons,numMultistart
        res = solve(nls, optimizer();maxiters=maxIterKing) 
    elseif NL_solve == :NonlinearSolve
        # Defining the objective function `f(x,p)`
        out = Vector{Equation}(undef,3nMod)
        # nls = NonlinearLeastSquaresProblem(fnls!, u0)
        if is_Jacobian
            if is_AD
            else
            end
            Js21!(JM, xx) = JacobCPEjL!(JM,xx,uhLN,L,nMod;nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,
                                    is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL)
            nls = NonlinearLeastSquaresProblem(NonlinearFunction(MhjL_GKMM!,jac==Js21!,resid_prototype = zeros(length(x0))), x0)      # , p
        else
            nls = NonlinearLeastSquaresProblem(NonlinearFunction(MhjL_GKMM!,resid_prototype = zeros(length(x0))), x0)
        end
        
        if Optlibary == :LeastSquaresOptim
            res = NonlinearSolve.solve(nls, LeastSquaresOptimJL(optimizer;linsolve=linsolve,autodiff=ADtype);maxiters=maxIterKing)
        elseif Optlibary == :NonlinearSolve  
        else
        end
    else
        edfbfbb
    end

    # @show res
    # @show res.retcode
    if NL_solve == :Optimization
        xfit = res.u         # the vector of best model1 parameters
        niter = res.stats.iterations
        is_converged = res.retcode
        xssr = res.objective                         # sum_kbn(abs2, fcur)
        # alg = res.alg
        # caches = res.cache
        # time = res.stats.time
        # fevals = res.stats.fevals
        # gevals = res.stats.gevals
        # hevals = res.stats.hevals
        # original = res.original
    elseif NL_solve == :NonlinearSolve
    else
    end
    # @show xssr, is_converged, xfit, niter
    return xssr, is_converged, xfit, niter
end

# nMode = 1
function optimMhjL(Mhst::AbstractVector{T},L::Int) where{T}

    uh2 = ((L+2.5) * (Mhst[2]) - (L+1.5) * Mhst[3])^0.5
    if iseven(L)
        uh = (uh2)^0.5
    else
        uh = sign(Mhst[1]) * (uh2)^0.5
    end
    vhth = (1/(T(L)+1.5) * (Mhst[2])^2 - uh2)^0.5
    nh = Mhst[1] / uh2 ^(L/2)
    return nh, uh, vhth
end

