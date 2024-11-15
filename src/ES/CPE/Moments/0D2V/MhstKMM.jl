
"""
  For weakly anisotropic and moderate anisotropic plasma in a degenerate state:

  The normalzied kinetic moments for local quasi-equilibrium state plasma 
    when the velocity space exhibits axisymmetry.
    When the `Lᵗʰ`-order amplitude of the normalized distribution function, `f̂ₗ(v̂)`,
    the `(j,L)ᵗʰ`-order normalzied kinetic moment can be expressed as:
  
        `Mhst(j,L) = 𝓜ⱼ(f̂ₗ) = 4π * ∫₀^∞(v̂ʲ⁺² * f̂ₗ) dv̂, j ≥ -L-2`.
  
    When `f̂ₗ(v̂)` is approximated by the KMM, any order normalzied kinetic moment can be obtained in a closed-form.

  I. Theory formulations

    1. When `j ∈ {(2jₚ - L - 2) | jₚ ∈ [0,N⁺]}`, the normalzied kinetic moment will be:

      `𝓜ⱼ(f̂₀) = CMjL * ∑ᵣ₌₁ᴺᴷ{n̂ₐᵣ*(v̂ₐₜₕᵣ)ʲ * (ûₐᵣ/v̂ₐₜₕᵣ)ᴸ * [1 + ∑ₖ₌₁^N CjLk * (ûₐᵣ/v̂ₐₜₕᵣ)²ᵇ]}.`
        
    where

      `N = j/2,     L ∈ 2N⁺ - 2,
           (j-1)/2, L ∈ 2N⁺ - 1`
      `CMjL = (j+L+1)!! / (2L-1)!! * 2^((j-L)/2)`
      `CjLk = (2L+1)!! / (2(L+k)+1)!! * C((j-L)/2,k)`

    Here, `C((j-L)/2,k)` is the binomial coefficient `Cₙᵏ` when `n=(j-L)/2`.

    2. When `-L-2 ≤ j ≤ L-2`, 
      2.1 `j ∈ 2ℕ`, the normalzied kinetic moment will be:
      
      2.2 `j ∈ 2ℕ+1`, the normalzied kinetic moment will be:

  II. Approximated formulations by utilizing Taylor expansion at `ûₐᵣ/v̂ₐₜₕᵣ = 0` 
      for weakly anisotropic and moderate anisotropic plasma where `ûₐᵣ/v̂ₐₜₕᵣ ≤ 1`.
      the approximated normalzied kinetic moment can be expressed as:

      `𝓜ⱼ(f̂₀) = cMjL * ∑ᵣ₌₁ᴺᴷ{n̂ₐᵣ*(v̂ₐₜₕᵣ)ʲ * (ûₐᵣ/v̂ₐₜₕᵣ)ᴸ * [1 + ∑ₖ₌₁^N cjLk * (ûₐᵣ/v̂ₐₜₕᵣ)²ᵇ]}.`
      
      where
       
       `cMjl = `
       `cjLk = `


  If `is_renorm == true`,
    `𝓜ⱼ(f̂₀) /= CMjL  , j ∈ {(2jₚ - L - 2) | jₚ ∈ [0,N⁺]}`
  end
  
  Notes: `{𝓜₁}/3 = Î ≠ û`, generally. Only when `nMod = 1` gives `Î = û`.
  
  See Ref. of Wang (2025) titled as "Relaxation model for a homogeneous plasma with axisymmetric velocity space".

"""

"""

  Inputs:
    L: 
    jvec: employing `∀(j,dj)`
    uai: = ûₐᵣ = uₐᵣ / vₐₜₕ
    mathtype: ∈ [:Exact, :Taylor0, :Taylor1, :TaylorInf]

  Outputs
    Mhst = MhsKMM!(Mhst,jvec,L,nai,uai,vthi,uhLN,nMod,ns;is_renorm=is_renorm,is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL,mathtype=mathtype)
    Mhst = MhsKMM!(Mhst,jvec,L,uai,uhLN,ns;is_renorm=is_renorm,is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL,mathtype=mathtype)

"""

# 2.5D, [nMod,njMs,ns]
function MhsKMM!(Mhst::AbstractArray{T},jvec::Vector{Int},L::Int,
    nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},
    vthi::Vector{AbstractVector{T}},uhLN::AbstractVector{T},
    nMod::Vector{Int},ns::Int64;is_renorm::Bool=true,is_norm_uhL::Bool=true,
    rtol_OrjL::T=1e-10,mathtype::Symbol=:Exact) where{T}
    
    for isp in 1:ns 
        if nMod == 1
            MhsKMM!(Mhst[:,isp],jvec,L,uai[isp][vec],uhLN[isp];
                    is_renorm=is_renorm,is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL,mathtype=mathtype) 
        else
            vec = 1:nMod[isp] 
            MhsKMM!(Mhst[:,isp],jvec,L,nai[isp][vec],uai[isp][vec],vthi[isp][vec],uhLN[isp],nMod[isp];
                    is_renorm=is_renorm,is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL,mathtype=mathtype) 
        end
    end
    return Mhst
end

# 2.5D, [nMod,njMs,LM]
function MhsKMM!(Mhst::AbstractArray{T},jvec::Vector{Int},LM::Int,
    nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},
    vthi::Vector{AbstractVector{T}},uhLN::AbstractVector{T},
    nMod::Vector{Int};is_renorm::Bool=true,is_norm_uhL::Bool=true,
    rtol_OrjL::T=1e-10,mathtype::Symbol=:Exact) where{T}
    
    for L in 0:LM
        if nMod == 1
            MhsKMM!(Mhst[:,isp],jvec,L,uai[isp][vec],uhLN[isp];
                    is_renorm=is_renorm,is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL,mathtype=mathtype) 
        else
            vec = 1:nMod[isp] 
            MhsKMM!(Mhst[:,isp],jvec,L,nai[isp][vec],uai[isp][vec],vthi[isp][vec],uhLN[isp],nMod[isp];
                    is_renorm=is_renorm,is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL,mathtype=mathtype) 
        end
    end
    return Mhst
end
 
# 1.5D, [nMod,njMs]
function MhsKMM!(Mhst::AbstractVector{T},jvec::Vector{Int},L::Int,
    nai::AbstractVector{T},uai::AbstractVector{T},
    vthi::AbstractVector{T},uhLN::T,nMod::Int;
    is_renorm::Bool=true,is_norm_uhL::Bool=true,rtol_OrjL::T=1e-10,mathtype::Symbol=:Exact) where{T}
    
    if uhLN ≤ eps(T)
        if L == 0
            MhsMMM!(Mhst,jvec,nai,vthi;is_renorm=is_renorm) 
        else
            Mhst[:] .= 0.0
        end
    else
        k = 0 
        for j in jvec 
            k += 1
            Mhst[k] = MhsKMM(j,L,nai,uai,vthi,uhLN,nMod; 
                    is_renorm=is_renorm,is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL,mathtype=mathtype) 
        end
    end
    return Mhst
end

# 0.5D, [nMod]
function MhsKMM(j::Int,L::Int,nai::AbstractVector{T},uai::AbstractVector{T},
    vthi::AbstractVector{T},uhLN::T,nMod::Int;
    is_renorm::Bool=true,is_norm_uhL::Bool=true,rtol_OrjL::T=1e-10,mathtype::Symbol=:Exact) where{T}
    
    if uhLN ≤ eps(T)
        if L == 0
            return MhsMMM(j,nai,vthi;is_renorm=is_renorm) 
        else
            return 0.0 |> T
        end
    else
        Mh = 0.0 
        for s = 1:nMod 
            Mh += MhsKMM(j,L,nai[s],uai[s],vthi[s],uhLN;
                        is_renorm=true,is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL,mathtype=mathtype) 
        end
        if is_renorm
            return Mh
        else
            if j == L
                return CMLL(T(L)) * Mh
            else
                return CMjL(T(j),T(L)) * Mh
            end
        end
    end
end

"""
    Mhst = MhsKMM(j,L,nai,uai,vthi,uhLN;
                is_renorm=is_renorm,is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL,mathtype=mathtype)
    Mhst = MhsKMM(L,nai,uai,uhLN;is_renorm=is_renorm,is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL)
"""

# 0.5D, []
function MhsKMM(j::Int,L::Int,nai::T,uai::T,vthi::T,uhLN::T;
    is_renorm::Bool=true,is_norm_uhL::Bool=true,rtol_OrjL::T=1e-10,mathtype::Symbol=:Exact) where{T}
    
    if L == 0
        return MhsKMM0(j,nai,uai,vthi;
                       is_renorm=is_renorm,rtol_OrjL=rtol_OrjL,mathtype=mathtype)
    else
        if abs(uai) ≤ eps(T)
            return 0.0 |> T
        else
            if j == L
                return MhsKMM(L,nai,uai,uhLN;is_renorm=is_renorm,is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL)
            else
                jL = j-L
                if iseven(jL)
                    if isone(vthi)
                        if jL < 0
                            hhjmmm
                            if mathtype == :Exact
                            elseif mathtype == :Taylor0
                            else
                                ygtrttt
                                if mathtype == :Taylor1
                                elseif TaylorInf
                                else
                                    aswdfcgv
                                end
                            end
                        else
                            # uvth2 = (uai/vthi)^2
                            N = jL / 2 |> Int
                            OrjL = OrjLNb((uai)^2,j,L,N;rtol_OrjL=rtol_OrjL)
                            if L == 1
                                if is_norm_uhL
                                    if is_renorm
                                        return MhrjL0D2V(j,L,nai,uai/uhLN,OrjL)
                                    else
                                        return CMjL(T(j),T(L)) * MhrjL0D2V(j,L,nai,uai/uhLN,OrjL)
                                    end
                                else
                                    if is_renorm
                                        return MhrjL0D2V(j,L,nai,uai,OrjL)
                                    else
                                        return CMjL(T(j),T(L)) * MhrjL0D2V(j,L,nai,uai,OrjL)
                                    end
                                end
                            else
                                if is_norm_uhL
                                    if is_renorm
                                        return MhrjL0D2V(j,L,nai,(uai/uhLN)^L,OrjL)
                                    else
                                        return CMjL(T(j),T(L)) * MhrjL0D2V(j,L,nai,(uai/uhLN)^L,OrjL)
                                    end
                                else
                                    if is_renorm
                                        return MhrjL0D2V(j,L,nai,uai^L,OrjL)
                                    else
                                        return CMjL(T(j),T(L)) * MhrjL0D2V(j,L,nai,uai^L,OrjL)
                                    end
                                end
                            end
                        end
                    else
                        if jL < 0
                            hhjmmm
                            if mathtype == :Exact
                            elseif mathtype == :Taylor0
                            else
                                ygtrttt
                                if mathtype == :Taylor1
                                elseif TaylorInf
                                else
                                    aswdfcgv
                                end
                            end
                        else
                            # uvth2 = (uai/vthi)^2
                            N = jL / 2 |> Int
                            OrjL = OrjLNb((uai/vthi)^2,j,L,N;rtol_OrjL=rtol_OrjL)
                            # @show uai
                            if L == 1
                                if is_norm_uhL
                                    if is_renorm
                                        return MhrjL0D2V(j,L,nai,uai/uhLN,vthi^jL,OrjL)
                                    else
                                        return CMjL(T(j),T(L)) * MhrjL0D2V(j,L,nai,uai/uhLN,vthi^jL,OrjL)
                                    end
                                else
                                    if is_renorm
                                        return MhrjL0D2V(j,L,nai,uai,vthi^jL,OrjL)
                                    else
                                        return CMjL(T(j),T(L)) * MhrjL0D2V(j,L,nai,uai,vthi^jL,OrjL)
                                    end
                                end
                            else
                                if is_norm_uhL
                                    if is_renorm
                                        return MhrjL0D2V(j,L,nai,(uai/uhLN)^L,vthi^jL,OrjL)
                                    else
                                        return CMjL(T(j),T(L)) * MhrjL0D2V(j,L,nai,(uai/uhLN)^L,vthi^jL,OrjL)
                                    end
                                else
                                    if is_renorm
                                        return MhrjL0D2V(j,L,nai,uai^L,vthi^jL,OrjL)
                                    else
                                        return CMjL(T(j),T(L)) * MhrjL0D2V(j,L,nai,uai^L,vthi^jL,OrjL)
                                    end
                                end
                            end
                        end
                    end
                else
                    sdfgbhnm
                    if mathtype == :Exact
                    elseif mathtype == :Taylor0
                    else
                        ygtrttt
                        if mathtype == :Taylor1
                        elseif TaylorInf
                        else
                            aswdfcgv
                        end
                    end
                end
            end
        end
    end
end

# j = L
function MhsKMM(L::Int,nai::T,uai::T,uhLN::T;is_renorm::Bool=true,is_norm_uhL::Bool=true,rtol_OrjL::T=1e-10) where{T}
    
    if L == 0
        return MhsKMM0(nai)
    else
        if abs(uai) ≤ eps(T)
            return 0.0 |> T
        else
            if L == 1
                if is_norm_uhL
                    if is_renorm
                        return MhrLL0D2V(L,nai,uai/uhLN)
                    else
                        return CMLL(T(L)) * MhrLL0D2V(L,nai,uai/uhLN)
                    end
                else
                    if is_renorm
                        return MhrLL0D2V(L,nai,uai)
                    else
                        return CMLL(T(L)) * MhrLL0D2V(L,nai,uai)
                    end
                end
            else
                if is_norm_uhL
                    if is_renorm
                        return MhrLL0D2V(L,nai,(uai/uhLN)^L)
                    else
                        return CMLL(T(L)) * MhrLL0D2V(L,nai,(uai/uhLN)^L)
                    end
                else
                    if is_renorm
                        return MhrLL0D2V(L,nai,uai^L)
                    else
                        return CMLL(T(L)) * MhrLL0D2V(L,nai,uai^L)
                    end
                end
            end
        end
    end
end

"""
"""

# nMod = 1 -> nai = 1, vthi = 1

# 2D, [njMs,ns]
function MhsKMM!(Mhst::AbstractArray{T},jvec::Vector{Int},L::Int,uai::T,ns::Int64;
    is_renorm::Bool=true,is_norm_uhL::Bool=true,
    rtol_OrjL::T=1e-10,mathtype::Symbol=:Exact) where{T}

    for isp in 1:ns 
        MhsKMM!(Mhst[:,isp],jvec,L,uai;
                is_renorm=is_renorm,is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL,mathtype=mathtype) 
    end
    return Mhst
end

# 1D, [njMs]
function MhsKMM!(Mhst::AbstractVector{T},jvec::Vector{Int},L::Int,uai::T;
    is_renorm::Bool=true,is_norm_uhL::Bool=true,
    rtol_OrjL::T=1e-10,mathtype::Symbol=:Exact) where{T}

    k = 0 
    for j in jvec 
        Mhst[k+1] = MhsKMM(j,L,uai;is_renorm=is_renorm,is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL,mathtype=mathtype)
    end
    return Mhst
end

# 0D, []   
function MhsKMM(j::Int,L::Int,uai::T;is_renorm::Bool=true,is_norm_uhL::Bool=true,
    rtol_OrjL::T=1e-10,mathtype::Symbol=:Exact) where{T}
    
    if L == 0
        return MhsKMM0(j,uai;is_renorm=is_renorm,rtol_OrjL=rtol_OrjL,mathtype=mathtype)
    else
        if abs.(uai) ≤ eps(T)
            return 0.0 |> T
        else
            if j == L 
                return MhsKMM(L,uai;is_renorm=is_renorm,is_norm_uhL=is_norm_uhL,rtol_OrjL=rtol_OrjL)
            else
                jL = j-L
                if iseven(jL)
                    if jL < 0
                        hhjmmm
                        if mathtype == :Exact
                        elseif mathtype == :Taylor0
                        else
                            ygtrttt
                            if mathtype == :Taylor1
                            elseif TaylorInf
                            else
                                aswdfcgv
                            end
                        end
                    else
                        # uvth2 = (uai/vthi)^2
                        N = jL / 2 |> Int
                        OrjL = OrjLNb((uai)^2,j,L,N;rtol_OrjL=rtol_OrjL)
                        if L == 1
                            if is_renorm
                                return MhrjL0D2V(j,L,nai,uai,OrjL)
                            else
                                return CMjL(T(j),T(L)) * MhrjL0D2V(j,L,nai,uai,OrjL)
                            end
                        else
                            if is_renorm
                                return MhrjL0D2V(j,L,nai,uai^L,OrjL)
                            else
                                return CMjL(T(j),T(L)) * MhrjL0D2V(j,L,nai,uai^L,OrjL)
                            end
                        end
                    end
                else
                    sdfgbhnm
                    if mathtype == :Exact
                    elseif mathtype == :Taylor0
                    else
                        ygtrttt
                        if mathtype == :Taylor1
                        elseif TaylorInf
                        else
                            aswdfcgv
                        end
                    end
                end
            end
        end
    end
end

# j = L
function MhsKMM(L::Int,uai::T;is_renorm::Bool=true,is_norm_uhL::Bool=true,rtol_OrjL::T=1e-10) where{T}
    
    if L == 0
        return 1.0 |> T
    else
        if abs.(uai) ≤ eps(T)
            return 0.0 |> T
        else
            if L == 1
                if is_renorm
                    return MhrLL0D2V(L,uai)
                else
                    return CMLL(T(L)) * MhrLL0D2V(L,uai)
                end
            else
                if is_renorm
                    return MhrLL0D2V(L,uai^L)
                else
                    return CMLL(T(L)) * MhrLL0D2V(L,uai^L)
                end
            end
        end
    end
end


