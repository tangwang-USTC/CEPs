
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
    Mhst = MhsKMM!(Mhst,jvec,L,nai,uai,vthi,nMod,ns;is_renorm=is_renorm,mathtype=mathtype)
    Mhst = MhsKMM!(Mhst,jvec,L,uai,ns;is_renorm=is_renorm,mathtype=mathtype)

"""

# 2.5D, [nMod,njMs,ns]
function MhsKMM!(Mhst::AbstractArray{T},jvec::Vector{Int},L::Int,
    nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}},
    nMod::Vector{Int},ns::Int64;is_renorm::Bool=true,mathtype::Symbol=:Exact) where{T}
    
    for isp in 1:ns 
        if nMod == 1
            MhsKMM!(Mhst[:,isp],jvec,L,uai[isp][vec];is_renorm=is_renorm,mathtype=mathtype) 
        else
            vec = 1:nMod[isp] 
            MhsKMM!(Mhst[:,isp],jvec,L,nai[isp][vec],uai[isp][vec],vthi[isp][vec],nMod[isp];is_renorm=is_renorm,mathtype=mathtype) 
        end
    end
    return Mhst
end

# 2.5D, [nMod,njMs,LM]
function MhsKMM!(Mhst::AbstractArray{T},jvec::Vector{Int},LM::Int,
    nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}},
    nMod::Vector{Int};is_renorm::Bool=true,mathtype::Symbol=:Exact) where{T}
    
    for L in 0:LM
        if nMod == 1
            MhsKMM!(Mhst[:,isp],jvec,L,uai[isp][vec];is_renorm=is_renorm,mathtype=mathtype) 
        else
            vec = 1:nMod[isp] 
            MhsKMM!(Mhst[:,isp],jvec,L,nai[isp][vec],uai[isp][vec],vthi[isp][vec],nMod[isp];is_renorm=is_renorm,mathtype=mathtype) 
        end
    end
    return Mhst
end

# 1.5D, [nMod,njMs]
function MhsKMM!(Mhst::AbstractVector{T},jvec::Vector{Int},L::Int,
    nai::AbstractVector{T},uai::AbstractVector{T},vthi::AbstractVector{T},
    nMod::Int;is_renorm::Bool=true,mathtype::Symbol=:Exact) where{T}
    
    if sum(abs.(uai)) ≤ eps(T)
        MhsMMM!(Mhst,jvec,L,nai,vthi;is_renorm=is_renorm) 
    else
        k = 0 
        for j in jvec 
            k += 1
            Mhst[k] = MhsKMM(j,L,nai,uai,vthi,nMod;is_renorm=is_renorm,mathtype=mathtype) 
        end
    end
    return Mhst
end

# 0.5D, [nMod]
function MhsKMM(j::Int,L::Int,nai::AbstractVector{T},uai::AbstractVector{T},vthi::AbstractVector{T},
    nMod::Int;is_renorm::Bool=true,mathtype::Symbol=:Exact) where{T}
    
    if sum(abs.(uai)) ≤ eps(T)
        return MhsMMM(j,L,nai,vthi;is_renorm=is_renorm) 
    else
        Mh = 0.0 
        for s = 1:nMod 
            Mh += MhsKMM(j,L,nai[s],uai[s],vthi[s];is_renorm=true,mathtype=mathtype) 
        end
        if is_renorm
            return Mh
        else
            return CMjL(j) * Mh
        end
    end
end

# 0.5D, []
function MhsKMM(j::Int,L::Int,nai::T,uai::T,vthi::T;is_renorm::Bool=true,mathtype::Symbol=:Exact) where{T}
    
    if L == 0
    else
        hghhhh
    end
    if sum(abs.(uai)) ≤ eps(T)
        if isone(vthi)
            return MhsMMM(j;is_renorm=is_renorm) * nai
        else
            return MhsMMM(j;is_renorm=is_renorm) * (nai * vthi^j)
        end
    else
        if iseven(j)
            if isone(vthi)
                if j == -2
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
                    a = 1.0 
                    for k in 1:Int(j/2)
                        a += CjLk(j,k) * uai^(2k)
                    end
                end
                if is_renorm
                    return a * nai
                else
                    return CMjL(j) * a * nai
                end
            else
                if j == -2
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
                    a = 1.0
                    uhh = uai/vthi 
                    for k in 1:Int(j/2)
                        a += CjLk(j,k) * (uhh)^(2k)
                    end
                end
                if is_renorm
                    return a * nai * vthi^j
                else
                    return CMjL(j) * a * nai * vthi^j
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

"""
"""

# nMod = 1 -> nai = 1, vthi = 1

# 2D, [njMs,ns]
function MhsKMM!(Mhst::AbstractArray{T},jvec::Vector{Int},L::Int,
    uai::T,ns::Int64;is_renorm::Bool=true,mathtype::Symbol=:Exact) where{T}

    for isp in 1:ns 
        MhsKMM!(Mhst[:,isp],jvec,L,uai;is_renorm=is_renorm,mathtype=mathtype) 
    end
    return Mhst
end

# 1D, [njMs]
function MhsKMM!(Mhst::AbstractVector{T},jvec::Vector{Int},L::Int,
    uai::T;is_renorm::Bool=true,mathtype::Symbol=:Exact) where{T}

    k = 0 
    for j in jvec 
        Mhst[k+1] = MhsKMM(j,L,uai;is_renorm=is_renorm,mathtype=mathtype)
    end
    return Mhst
end

# 0D, []   
function MhsKMM(j::Int,L::Int,uai::T;is_renorm::Bool=true,mathtype::Symbol=:Exact) where{T}
    
    if L == 0
    else
        sdcfvbb
    end
    if abs(uai) ≤ eps(T)
        return MhsMMM(j)
    else
        if iseven(j)
            if j == -2
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
                a = 1.0 
                for k in 1:(j/2) 
                    a += CjLk(j,k) * uai^(2k)
                end
            end
            if is_renorm
                return a
            else
                return CMjL(j) * a
            end
        else
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



