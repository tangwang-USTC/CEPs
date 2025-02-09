

"""
  Applying Nonlinear Programming (NLP) method to solve the Characteristic parameter equations (CPEs) 
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
  CPEs

  Inputs:
    out: = zeros(nMod,nMod)
    Mhst: = M̂ⱼₗ*, which is the renormalized general kinetic moments.

  Outputs:
    res = CPEjL(x,uhLN,L,nMod;Mhst=Mhst,is_norm_uhL=is_norm_uhL)

""" 

# nMode ≥ 2
# For "Optimization.jl"
function CPEjL(x::AbstractVector{T}, uhLN::Tf, L::Int, nMod::Int;
    Mhst::AbstractVector{Tf}=[0.1, 1.0], is_norm_uhL::Bool=true) where{T,Tf}

    if L == 100000
    #     return CPEj0(x,nMod;Mhst=Mhst)
    # elseif L == 111
    #     return CPEj1(x,uhLN, nMod;Mhst=Mhst,is_norm_uhL=is_norm_uhL)
    else
        nh = x[1:3:end]
        uh = x[2:3:end]
        vhth = x[3:3:end]
    
        vhth2 = vhth.^2 
        uvth2 = (uh).^2 ./ vhth2              # uh .^ 2 ./ vhth .^ 2
        
        if is_norm_uhL
            uhL = (uh / uhLN).^L
        else
            uhL = uh.^L
        end
    
        nj = 1
        # j = L + 0
        out1 = sum(nh .* uhL) - Mhst[nj]
    
        nj += 1
        j = L + 2
        OrjL = CjLk(j,L,1) * uvth2
        out2 = sum((nh .* vhth2 .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
    
        nj += 1
        j = L + 4
        # OrjL = zeros(T,nMod)
        N = 2
        OrjL = OrjLNb(uvth2,j,L,N)
        out3 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]

        # k == 2
        nj += 1
        j += 2
        N = (j - L) / 2 |> Int
        OrjL = OrjLNb(uvth2,j,L,N)
        out4 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
        nj += 1
        j += 2
        N = (j - L) / 2 |> Int
        OrjL = OrjLNb(uvth2,j,L,N)
        out5 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
        nj += 1
        j += 2
        N = (j - L) / 2 |> Int
        OrjL = OrjLNb(uvth2,j,L,N)
        out6 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
        if nMod == 2
            return out1^2 + out2^2 + out3^2 + out4^2 + out5^2 + out6^2
        else
            nj += 1
            j += 2
            N = (j - L) / 2 |> Int
            OrjL = OrjLNb(uvth2,j,L,N)
            out7 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
            nj += 1
            j += 2
            N = (j - L) / 2 |> Int
            OrjL = OrjLNb(uvth2,j,L,N)
            out8 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
            nj += 1
            j += 2
            N = (j - L) / 2 |> Int
            OrjL = OrjLNb(uvth2,j,L,N)
            out9 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
            if nMod == 3
                return out1^2 + out2^2 + out3^2 + out4^2 + out5^2 + out6^2 + out7^2 + out8^2 + out9^2
            else
                if nMod == 4
                    nj += 1
                    j += 2
                    N = (j - L) / 2 |> Int
                    OrjL = OrjLNb(uvth2,j,L,N)
                    out10 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                    nj += 1
                    j += 2
                    N = (j - L) / 2 |> Int
                    OrjL = OrjLNb(uvth2,j,L,N)
                    out11 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                    nj += 1
                    j += 2
                    N = (j - L) / 2 |> Int
                    OrjL = OrjLNb(uvth2,j,L,N)
                    out12 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                else
                    if nMod == 5
                        return out1^2 + out2^2 + out3^2 + out4^2 + out5^2 + out6^2 + out7^2 + out8^2 + out9^2 + out10^2 + out11^2 + out12^2
                    else
                        nj += 1
                        j += 2
                        N = (j - L) / 2 |> Int
                        OrjL = OrjLNb(uvth2,j,L,N)
                        out13 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                        nj += 1
                        j += 2
                        N = (j - L) / 2 |> Int
                        OrjL = OrjLNb(uvth2,j,L,N)
                        out14 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                        nj += 1
                        j += 2
                        N = (j - L) / 2 |> Int
                        OrjL = OrjLNb(uvth2,j,L,N)
                        out15 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                        if nMod == 6
                            return out1^2 + out2^2 + out3^2 + out4^2 + out5^2 + out6^2 + out7^2 + out8^2 + out9^2 + out10^2 + out11^2 + out12^2 + out13^2 + out14^2 + out15^2
                        else
                            duuuuuifff
                            if k in 2:nMod
                                for s in 1:3
                                    nj += 1
                                    j += 2
                                    N = (j - L) / 2 |> Int
                                    OrjL = OrjLNb(uvth2,j,L,N)
                                    out[nj] = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                                end
                            end
                            @show out
                            return sum(abs2,out)
                        end
                    end
                end
            end
        end
    end
end
 
# For "Optimization.jl" and `is_constraint=true`
function CPEjL(x::AbstractVector{T}, uhLN::Tf, L::Int, nMod::Int, Ncons::Int;
    Mhst::AbstractVector{Tf}=[0.1, 1.0], is_norm_uhL::Bool=true) where{T,Tf}

    if L == 100000
    #     return CPEj0(x,nMod;Mhst=Mhst)
    # elseif L == 111
    #     return CPEj1(x,uhLN, nMod;Mhst=Mhst,is_norm_uhL=is_norm_uhL)
    else
        nh = x[1:3:end]
        uh = x[2:3:end]
        vhth = x[3:3:end]
    
        vhth2 = vhth.^2 
        uvth2 = (uh).^2 ./ vhth2              # uh .^ 2 ./ vhth .^ 2
        
        if is_norm_uhL
            uhL = (uh / uhLN).^L
        else
            uhL = uh.^L
        end
    

        if Ncons == 3
            return CPEjLN3(x,uhLN,L,nMod;Mhst=Mhst,is_norm_uhL=is_norm_uhL)
        elseif Ncons == 2
            return CPEjLN2(x,uhLN,L,nMod;Mhst=Mhst,is_norm_uhL=is_norm_uhL)
        elseif Ncons == 1
            return CPEjLN1(x,uhLN,L,nMod;Mhst=Mhst,is_norm_uhL=is_norm_uhL)
        elseif Ncons == 0
            return CPEjLN0(x,uhLN,L,nMod;Mhst=Mhst,is_norm_uhL=is_norm_uhL)
        else
            nj = 1
            # j = L + 0
            out1 = sum(nh .* uhL) - Mhst[nj]
        
            nj += 1
            j = L + 2
            OrjL = CjLk(j,L,1) * uvth2
            out2 = sum((nh .* vhth2 .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
        
            nj += 1
            j = L + 4
            N = 2
            CjLks = CjLk(T(j),T(L),T.(1:N))
            OrjL =  CjLks[1] * uvth2 + CjLks[2] * uvth2 .^2
            out3 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]

            # k == 2
            nj += 1
            j += 2
            N = (j - L) / 2 |> Int
            OrjL = OrjLNb(uvth2,j,L,N)
            out4 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
    
            nj += 1
            j += 2
            N = (j - L) / 2 |> Int
            OrjL = OrjLNb(uvth2,j,L,N)
            out5 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
            nj += 1
            j += 2
            N = (j - L) / 2 |> Int
            OrjL = OrjLNb(uvth2,j,L,N)
            out6 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
            if nMod == 2
                return out1^2 + out2^2 + out3^2 + out4^2 + out5^2 + out6^2
            else
                nj += 1
                j += 2
                N = (j - L) / 2 |> Int
                OrjL = OrjLNb(uvth2,j,L,N)
                out7 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                nj += 1
                j += 2
                N = (j - L) / 2 |> Int
                OrjL = OrjLNb(uvth2,j,L,N)
                out8 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                nj += 1
                j += 2
                N = (j - L) / 2 |> Int
                OrjL = OrjLNb(uvth2,j,L,N)
                out9 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                if nMod == 3
                    return out1^2 + out2^2 + out3^2 + out4^2 + out5^2 + out6^2 + out7^2 + out8^2 + out9^2
                else
                    if nMod == 4
                        nj += 1
                        j += 2
                        N = (j - L) / 2 |> Int
                        OrjL = OrjLNb(uvth2,j,L,N)
                        out10 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                        nj += 1
                        j += 2
                        N = (j - L) / 2 |> Int
                        OrjL = OrjLNb(uvth2,j,L,N)
                        out11 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                        nj += 1
                        j += 2
                        N = (j - L) / 2 |> Int
                        OrjL = OrjLNb(uvth2,j,L,N)
                        out12 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                    else
                        if nMod == 5
                            return out1^2 + out2^2 + out3^2 + out4^2 + out5^2 + out6^2 + out7^2 + out8^2 + out9^2 + out10^2 + out11^2 + out12^2
                        else
                            nj += 1
                            j += 2
                            N = (j - L) / 2 |> Int
                            OrjL = OrjLNb(uvth2,j,L,N)
                            out13 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                            nj += 1
                            j += 2
                            N = (j - L) / 2 |> Int
                            OrjL = OrjLNb(uvth2,j,L,N)
                            out14 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                            nj += 1
                            j += 2
                            N = (j - L) / 2 |> Int
                            OrjL = OrjLNb(uvth2,j,L,N)
                            out15 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                            if nMod == 6
                                return out1^2 + out2^2 + out3^2 + out4^2 + out5^2 + out6^2 + out7^2 + out8^2 + out9^2 + out10^2 + out11^2 + out12^2 + out13^2 + out14^2 + out15^2
                            else
                                duuuuuifff
                                if k in 2:nMod
                                    for s in 1:3
                                        nj += 1
                                        j += 2
                                        N = (j - L) / 2 |> Int
                                        OrjL = OrjLNb(uvth2,j,L,N)
                                        out[nj] = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                                    end
                                end
                                @show out
                                return sum(abs2,out)
                            end
                        end
                    end
                end
            end
        end
    end
end

function CPEjLN0(x::AbstractVector{T}, uhLN::Tf, L::Int, nMod::Int;
    Mhst::AbstractVector{Tf}=[0.1, 1.0], is_norm_uhL::Bool=true) where{T,Tf}

    if L == 100000
    #     return CPEj0(x,nMod;Mhst=Mhst)
    # elseif L == 111
    #     return CPEj1(x,uhLN, nMod;Mhst=Mhst,is_norm_uhL=is_norm_uhL)
    else
        nh = x[1:3:end]
        uh = x[2:3:end]
        vhth = x[3:3:end]
    
        vhth2 = vhth.^2 
        uvth2 = (uh).^2 ./ vhth2              # uh .^ 2 ./ vhth .^ 2
        
        if is_norm_uhL
            uhL = (uh / uhLN).^L
        else
            uhL = uh.^L
        end
    
        nj = 1
        # j = L + 0
        out1 = sum(nh .* uhL) - Mhst[nj]
    
        nj += 1
        j = L + 2
        OrjL = CjLk(j,L,1) * uvth2
        out2 = sum((nh .* vhth2 .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
    
        nj += 1
        j = L + 4
        N = 2
        CjLks = CjLk(T(j),T(L),T.(1:N))
        OrjL =  CjLks[1] * uvth2 + CjLks[2] * uvth2 .^2
        out3 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]

        # k == 2
        nj += 1
        j += 2
        N = (j - L) / 2 |> Int
        OrjL = OrjLNb(uvth2,j,L,N)
        out4 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]

        nj += 1
        j += 2
        N = (j - L) / 2 |> Int
        OrjL = OrjLNb(uvth2,j,L,N)
        out5 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
        nj += 1
        j += 2
        N = (j - L) / 2 |> Int
        OrjL = OrjLNb(uvth2,j,L,N)
        out6 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
        if nMod == 2
            return out1^2 + out2^2 + out3^2 + out4^2 + out5^2 + out6^2
        else
            nj += 1
            j += 2
            N = (j - L) / 2 |> Int
            OrjL = OrjLNb(uvth2,j,L,N)
            out7 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
            nj += 1
            j += 2
            N = (j - L) / 2 |> Int
            OrjL = OrjLNb(uvth2,j,L,N)
            out8 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
            nj += 1
            j += 2
            N = (j - L) / 2 |> Int
            OrjL = OrjLNb(uvth2,j,L,N)
            out9 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
            if nMod == 3
                return out1^2 + out2^2 + out3^2 + out4^2 + out5^2 + out6^2 + out7^2 + out8^2 + out9^2
            else
                if nMod == 4
                    nj += 1
                    j += 2
                    N = (j - L) / 2 |> Int
                    OrjL = OrjLNb(uvth2,j,L,N)
                    out10 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                    nj += 1
                    j += 2
                    N = (j - L) / 2 |> Int
                    OrjL = OrjLNb(uvth2,j,L,N)
                    out11 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                    nj += 1
                    j += 2
                    N = (j - L) / 2 |> Int
                    OrjL = OrjLNb(uvth2,j,L,N)
                    out12 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                else
                    if nMod == 5
                        return out1^2 + out2^2 + out3^2 + out4^2 + out5^2 + out6^2 + out7^2 + out8^2 + out9^2 + out10^2 + out11^2 + out12^2
                    else
                        nj += 1
                        j += 2
                        N = (j - L) / 2 |> Int
                        OrjL = OrjLNb(uvth2,j,L,N)
                        out13 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                        nj += 1
                        j += 2
                        N = (j - L) / 2 |> Int
                        OrjL = OrjLNb(uvth2,j,L,N)
                        out14 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                        nj += 1
                        j += 2
                        N = (j - L) / 2 |> Int
                        OrjL = OrjLNb(uvth2,j,L,N)
                        out15 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                        if nMod == 6
                            return out1^2 + out2^2 + out3^2 + out4^2 + out5^2 + out6^2 + out7^2 + out8^2 + out9^2 + out10^2 + out11^2 + out12^2 + out13^2 + out14^2 + out15^2
                        else
                            duuuuuifff
                            if k in 2:nMod
                                for s in 1:3
                                    nj += 1
                                    j += 2
                                    N = (j - L) / 2 |> Int
                                    OrjL = OrjLNb(uvth2,j,L,N)
                                    out[nj] = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                                end
                            end
                            @show out
                            return sum(abs2,out)
                        end
                    end
                end
            end
        end
    end
end

function CPEjLN1(x::AbstractVector{T}, uhLN::Tf, L::Int, nMod::Int;
    Mhst::AbstractVector{Tf}=[0.1, 1.0], is_norm_uhL::Bool=true) where{T,Tf}

    if L == 100000
    #     return CPEj0(x,nMod;Mhst=Mhst)
    # elseif L == 111
    #     return CPEj1(x,uhLN, nMod;Mhst=Mhst,is_norm_uhL=is_norm_uhL)
    else
        nh = x[1:3:end]
        uh = x[2:3:end]
        vhth = x[3:3:end]
    
        vhth2 = vhth.^2 
        uvth2 = (uh).^2 ./ vhth2              # uh .^ 2 ./ vhth .^ 2
        
        if is_norm_uhL
            uhL = (uh / uhLN).^L
        else
            uhL = uh.^L
        end

        nj = 1
        j = L + 2
        OrjL = CjLk(j,L,1) * uvth2
        out2 = sum((nh .* vhth2 .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
    
        nj += 1
        j = L + 4
        N = 2
        
        CjLks = CjLk(T(j),T(L),T.(1:N))
        OrjL =  CjLks[1] * uvth2 + CjLks[2] * uvth2 .^2

        out3 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]

        # k == 2
        nj += 1
        j += 2
        N = (j - L) / 2 |> Int
        OrjL = OrjLNb(uvth2,j,L,N)
        out4 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]

        nj += 1
        j += 2
        N = (j - L) / 2 |> Int
        OrjL = OrjLNb(uvth2,j,L,N)
        out5 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
        nj += 1
        j += 2
        N = (j - L) / 2 |> Int
        OrjL = OrjLNb(uvth2,j,L,N)
        out6 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
        if nMod == 2
            return out2^2 + out3^2 + out4^2 + out5^2 + out6^2
        else
            nj += 1
            j += 2
            N = (j - L) / 2 |> Int
            OrjL = OrjLNb(uvth2,j,L,N)
            out7 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
            nj += 1
            j += 2
            N = (j - L) / 2 |> Int
            OrjL = OrjLNb(uvth2,j,L,N)
            out8 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
            nj += 1
            j += 2
            N = (j - L) / 2 |> Int
            OrjL = OrjLNb(uvth2,j,L,N)
            out9 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
            if nMod == 3
                return out2^2 + out3^2 + out4^2 + out5^2 + out6^2 + out7^2 + out8^2 + out9^2
            else
                if nMod == 4
                    nj += 1
                    j += 2
                    N = (j - L) / 2 |> Int
                    OrjL = OrjLNb(uvth2,j,L,N)
                    out10 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                    nj += 1
                    j += 2
                    N = (j - L) / 2 |> Int
                    OrjL = OrjLNb(uvth2,j,L,N)
                    out11 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                    nj += 1
                    j += 2
                    N = (j - L) / 2 |> Int
                    OrjL = OrjLNb(uvth2,j,L,N)
                    out12 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                else
                    if nMod == 5
                        return out2^2 + out3^2 + out4^2 + out5^2 + out6^2 + out7^2 + out8^2 + out9^2 + out10^2 + out11^2 + out12^2
                    else
                        nj += 1
                        j += 2
                        N = (j - L) / 2 |> Int
                        OrjL = OrjLNb(uvth2,j,L,N)
                        out13 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                        nj += 1
                        j += 2
                        N = (j - L) / 2 |> Int
                        OrjL = OrjLNb(uvth2,j,L,N)
                        out14 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                        nj += 1
                        j += 2
                        N = (j - L) / 2 |> Int
                        OrjL = OrjLNb(uvth2,j,L,N)
                        out15 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                        if nMod == 6
                            return out2^2 + out3^2 + out4^2 + out5^2 + out6^2 + out7^2 + out8^2 + out9^2 + out10^2 + out11^2 + out12^2 + out13^2 + out14^2 + out15^2
                        else
                            duuuuuifff
                            if k in 2:nMod
                                for s in 1:3
                                    nj += 1
                                    j += 2
                                    N = (j - L) / 2 |> Int
                                    OrjL = OrjLNb(uvth2,j,L,N)
                                    out[nj] = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                                end
                            end
                            @show out
                            return sum(abs2,out)
                        end
                    end
                end
            end
        end
    end
end

function CPEjLN2(x::AbstractVector{T}, uhLN::Tf, L::Int, nMod::Int;
    Mhst::AbstractVector{Tf}=[0.1, 1.0], is_norm_uhL::Bool=true) where{T,Tf}

    if L == 100000
    #     return CPEj0(x,nMod;Mhst=Mhst)
    # elseif L == 111
    #     return CPEj1(x,uhLN, nMod;Mhst=Mhst,is_norm_uhL=is_norm_uhL)
    else
        nh = x[1:3:end]
        uh = x[2:3:end]
        vhth = x[3:3:end]
    
        vhth2 = vhth.^2 
        uvth2 = (uh).^2 ./ vhth2              # uh .^ 2 ./ vhth .^ 2
        
        if is_norm_uhL
            uhL = (uh / uhLN).^L
        else
            uhL = uh.^L
        end

        nj = 1
        j = L + 4
        N = 2
        
        CjLks = CjLk(T(j),T(L),T.(1:N))
        OrjL =  CjLks[1] * uvth2 + CjLks[2] * uvth2 .^2

        out3 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]

        # k == 2
        nj += 1
        j += 2
        N = (j - L) / 2 |> Int
        OrjL = OrjLNb(uvth2,j,L,N)
        out4 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]

        nj += 1
        j += 2
        N = (j - L) / 2 |> Int
        OrjL = OrjLNb(uvth2,j,L,N)
        out5 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
        nj += 1
        j += 2
        N = (j - L) / 2 |> Int
        OrjL = OrjLNb(uvth2,j,L,N)
        out6 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
        if nMod == 2
            return out3^2 + out4^2 + out5^2 + out6^2
        else
            nj += 1
            j += 2
            N = (j - L) / 2 |> Int
            OrjL = OrjLNb(uvth2,j,L,N)
            out7 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
            nj += 1
            j += 2
            N = (j - L) / 2 |> Int
            OrjL = OrjLNb(uvth2,j,L,N)
            out8 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
            nj += 1
            j += 2
            N = (j - L) / 2 |> Int
            OrjL = OrjLNb(uvth2,j,L,N)
            out9 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
            if nMod == 3
                return out3^2 + out4^2 + out5^2 + out6^2 + out7^2 + out8^2 + out9^2
            else
                if nMod == 4
                    nj += 1
                    j += 2
                    N = (j - L) / 2 |> Int
                    OrjL = OrjLNb(uvth2,j,L,N)
                    out10 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                    nj += 1
                    j += 2
                    N = (j - L) / 2 |> Int
                    OrjL = OrjLNb(uvth2,j,L,N)
                    out11 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                    nj += 1
                    j += 2
                    N = (j - L) / 2 |> Int
                    OrjL = OrjLNb(uvth2,j,L,N)
                    out12 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                else
                    if nMod == 5
                        return out3^2 + out4^2 + out5^2 + out6^2 + out7^2 + out8^2 + out9^2 + out10^2 + out11^2 + out12^2
                    else
                        nj += 1
                        j += 2
                        N = (j - L) / 2 |> Int
                        OrjL = OrjLNb(uvth2,j,L,N)
                        out13 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                        nj += 1
                        j += 2
                        N = (j - L) / 2 |> Int
                        OrjL = OrjLNb(uvth2,j,L,N)
                        out14 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                        nj += 1
                        j += 2
                        N = (j - L) / 2 |> Int
                        OrjL = OrjLNb(uvth2,j,L,N)
                        out15 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                        if nMod == 6
                            return out3^2 + out4^2 + out5^2 + out6^2 + out7^2 + out8^2 + out9^2 + out10^2 + out11^2 + out12^2 + out13^2 + out14^2 + out15^2
                        else
                            duuuuuifff
                            if k in 2:nMod
                                for s in 1:3
                                    nj += 1
                                    j += 2
                                    N = (j - L) / 2 |> Int
                                    OrjL = OrjLNb(uvth2,j,L,N)
                                    out[nj] = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                                end
                            end
                            @show out
                            return sum(abs2,out)
                        end
                    end
                end
            end
        end
    end
end

function CPEjLN3(x::AbstractVector{T}, uhLN::Tf, L::Int, nMod::Int;
    Mhst::AbstractVector{Tf}=[0.1, 1.0], is_norm_uhL::Bool=true) where{T,Tf}

    if L == 100000
    #     return CPEj0(x,nMod;Mhst=Mhst)
    # elseif L == 111
    #     return CPEj1(x,uhLN, nMod;Mhst=Mhst,is_norm_uhL=is_norm_uhL)
    else
        nh = x[1:3:end]
        uh = x[2:3:end]
        vhth = x[3:3:end]
    
        vhth2 = vhth.^2 
        uvth2 = (uh).^2 ./ vhth2              # uh .^ 2 ./ vhth .^ 2
        
        if is_norm_uhL
            uhL = (uh / uhLN).^L
        else
            uhL = uh.^L
        end
    
        # nj = 0
        # j = L + 4

        # # k == 2
        # nj += 1
        # j += 2

        nj = 1
        j = L + 6
        N = (j - L) / 2 |> Int
        OrjL = OrjLNb(uvth2,j,L,N)
        out4 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]

        nj += 1
        j += 2
        N = (j - L) / 2 |> Int
        OrjL = OrjLNb(uvth2,j,L,N)
        out5 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
        nj += 1
        j += 2
        N = (j - L) / 2 |> Int
        OrjL = OrjLNb(uvth2,j,L,N)
        out6 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
        if nMod == 2
            return out4^2 + out5^2 + out6^2
        else
            nj += 1
            j += 2
            N = (j - L) / 2 |> Int
            OrjL = OrjLNb(uvth2,j,L,N)
            out7 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
            nj += 1
            j += 2
            N = (j - L) / 2 |> Int
            OrjL = OrjLNb(uvth2,j,L,N)
            out8 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
            nj += 1
            j += 2
            N = (j - L) / 2 |> Int
            OrjL = OrjLNb(uvth2,j,L,N)
            out9 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
            if nMod == 3
                return out4^2 + out5^2 + out6^2 + out7^2 + out8^2 + out9^2
            else
                if nMod == 4
                    nj += 1
                    j += 2
                    N = (j - L) / 2 |> Int
                    OrjL = OrjLNb(uvth2,j,L,N)
                    out10 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                    nj += 1
                    j += 2
                    N = (j - L) / 2 |> Int
                    OrjL = OrjLNb(uvth2,j,L,N)
                    out11 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                    nj += 1
                    j += 2
                    N = (j - L) / 2 |> Int
                    OrjL = OrjLNb(uvth2,j,L,N)
                    out12 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                else
                    if nMod == 5
                        return out4^2 + out5^2 + out6^2 + out7^2 + out8^2 + out9^2 + out10^2 + out11^2 + out12^2
                    else
                        nj += 1
                        j += 2
                        N = (j - L) / 2 |> Int
                        OrjL = OrjLNb(uvth2,j,L,N)
                        out13 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                        nj += 1
                        j += 2
                        N = (j - L) / 2 |> Int
                        OrjL = OrjLNb(uvth2,j,L,N)
                        out14 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                        nj += 1
                        j += 2
                        N = (j - L) / 2 |> Int
                        OrjL = OrjLNb(uvth2,j,L,N)
                        out15 = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                        if nMod == 6
                            return out4^2 + out5^2 + out6^2 + out7^2 + out8^2 + out9^2 + out10^2 + out11^2 + out12^2 + out13^2 + out14^2 + out15^2
                        else
                            duuuuuifff
                            if k in 2:nMod
                                for s in 1:3
                                    nj += 1
                                    j += 2
                                    N = (j - L) / 2 |> Int
                                    OrjL = OrjLNb(uvth2,j,L,N)
                                    out[nj] = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
                                end
                            end
                            @show out
                            return sum(abs2,out)
                        end
                    end
                end
            end
        end
    end
end

# nMode = 1


"""
  CPEs

  Inputs:
    out: = zeros(nMod,nMod)
    Mhst: = M̂ⱼₗ*, which is the renormalized general kinetic moments.

  Outputs:
    res = CPEjL(x,uhLN,L,nMod,Mhst;out=out,nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL)

""" 

## Verification
function CPEjL(x::AbstractVector{T}, uhLN::T, L::Int, nMod::Int, Mhst::AbstractVector{T};
    out::AbstractVector{T}=[0.1, 1.0], nh::AbstractVector{T}=[0.1, 1.0], uh::AbstractVector{T}=[0.1, 1.0], 
    vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], 
    uvth2::AbstractVector{T}=[0.1, 1.0], is_norm_uhL::Bool=true,rtol_OrjL::T=1e-10) where{T}

    if L == 100000
        return CPEj0(x,nMod;out=out,nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,rtol_OrjL=rtol_OrjL)
    elseif L == 111
        return CPEj1(x,uhLN, nMod;out=out,nh=nh,uh=uh,vhth=vhth,vhth2=vhth2,uvth2=uvth2,Mhst=Mhst,is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL)
    else
        nh[:] = x[1:3:end]
        uh[:] = x[2:3:end]
        vhth[:] = x[3:3:end]
    
        vhth2[:] = vhth.^2 
        uvth2[:] = (uh).^2 ./ vhth2              # uh .^ 2 ./ vhth .^ 2

        uhLN = uhLNorm(nh,uh,L)
        if is_norm_uhL
            uhL = (uh / uhLN).^L
        else
            uhL = uh.^L
        end
    
        nj = 1
        # j = L + 0
        out[nj] = sum(nh .* uhL) - Mhst[nj]
    
        nj += 1
        j = L + 2
        OrjL = CjLk(T(j),T(L),T(1)) * uvth2
        out[nj] = sum((nh .* vhth2 .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
    
        nj += 1
        j = L + 4
        # OrjL = zeros(T,nMod)
        N = 2
        OrjLNb!(OrjL,uvth2,j,L,N,nMod;rtol_OrjL=rtol_OrjL)
        out[nj] = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
    
        for k in 2:nMod
            for s in 1:3
                nj += 1
                j += 2
                N = (j - L) / 2 |> Int
                OrjLNb!(OrjL,uvth2,j,L,N,nMod;rtol_OrjL=rtol_OrjL)
                out[nj] = sum((nh .* vhth2.^N .* uhL) .* (1 .+ OrjL)) - Mhst[nj]
            end
        end
        return sum(abs2,out)
    end
end


