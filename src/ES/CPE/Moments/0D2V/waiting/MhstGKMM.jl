
"""
  For weakly anisotropic and moderate anisotropic plasma in a non-degenerate state:

  The normalzied kinetic moments for local sub-equilibrium state plasma 
    when the velocity space exhibits axisymmetry.
    When the `Lᵗʰ`-order amplitude of the normalized distribution function, `f̂ₗ(v̂)`,
    the `(j,L)ᵗʰ`-order normalzied kinetic moment can be expressed as:
  
        `Mhst(j,L) = 𝓜ⱼ(f̂ₗ) = 4π * ∫₀^∞(v̂ʲ⁺² * f̂ₗ) dv̂, j ≥ -L-2`.
  
  When `f̂ₗ(v̂)` is approximated by the GKMM and employing `∀(j,dj)`, 
  the normalzied kinetic moment will be:

    `          CMjL * ∑ᵣ₌₁ᴺᴷ{n̂ₐₗᵣ*(v̂ₐₜₕₗᵣ)ʲ * [1 + ∑ₖ₌₁^(j/2) CjLk * (ûₐₗᵣ/v̂ₐₜₕₗᵣ)²ᵇ]} , L ∈ [0,2N⁺],
     𝓜ⱼ(f̂₀) = 
               CMjL * ∑ᵣ₌₁ᴺᴷ{n̂ₐₗᵣ*(v̂ₐₜₕₗᵣ)ʲ * (ûₐₗᵣ/v̂ₐₜₕₗᵣ)ᴸ * [1 + ∑ₖ₌₁^((j-1)/2) CjLk * (ûₐₗᵣ/v̂ₐₜₕₗᵣ)²ᵇ]} , L ∈ 2N⁺ - 1.`
               
  where
    
    `j ∈ {(2jₚ - L - 2) | jₚ ∈ [0,N⁺]}`
    `CMjL = (j+L+1)!! / (2L-1)!! * 2^((j-L)/2)`
    `CjLk = (2L+1)!! / (2(L+k)+1)!! * C((j-L)/2,k)`

  Here, `C((j-L)/2,k)` is the binomial coefficient `Cₙᵏ` when `n=(j-L)/2`.

  If `is_renorm == true`,
    `𝓜ⱼ(f̂₀) /= CMjL  , j ∈ {(2jₚ - L - 2) | jₚ ∈ [0,N⁺]}`
  end
  
  Notes: `{𝓜₁}/3 = Î ≠ û`, generally. Only when `nMod = 1` gives `Î = û`.
  
  See Ref. of Wang (2025) titled as "Relaxation model for a homogeneous plasma with axisymmetric velocity space".

  Inputs:
    jvec:
    uai: = ûₐₗᵣ = uₐₗᵣ / vₐₜₕ;

  Outputs
    Mhst = MhsKMM!(Mhst,jvec,nai,uai,vthi,nMod,ns;is_renorm=is_renorm,mathtype=mathtype)
    Mhst = MhsKMM!(Mhst,jvec,uai,ns;is_renorm=is_renorm,mathtype=mathtype)

"""
