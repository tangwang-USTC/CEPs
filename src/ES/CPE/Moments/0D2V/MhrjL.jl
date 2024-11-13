

"""
  The renormalized kinetic moment of sub-component of amplitude function:
  
    `𝓜ᵣⱼₗ := 𝓜ᵣⱼ(f̂ₗ) = MhrjL`

    `MhrjL := n̂ᵣₗ * (ûᵣₗ)ᴸ * (v̂thᵣₗ)ʲ⁻ᴸ * (1 + OrnL)`

  where `OrnL = Oᵣⱼₗᴺᵇ`

  Inputs:
  Outputs:
    MhrjL = MhrjL0D2V(j,L,nh,uhL,vhthjL,OrnL)
"""

function MhrjL0D2V(j::Int, L::Int, nh::T, uhL::T, vhthjL::T, OrnL::T) where {T <: Real}
  
  if j == L
    return MhrLL0D2V(L,nh,uhL)
  else
    if L == 0
      return Mhrj00D2V(j,nh,vhthjL,OrnL)
    else
      return nh * uhL * vhthjL * (1 + OrnL)
    end
  end
end


function MhrjL0D2V(j::Int, L::Int, nh::T, uhL::T, OrnL::T) where {T <: Real}
  
  if j == L
    return MhrLL0D2V(L,nh,uhL)
  else
    if L == 0
      return Mhrj00D2V(j,nh,OrnL)
    else
      return nh * uhL * (1 + OrnL)
    end
  end
end

"""
  When `L == 0`

  Inputs:
  Outputs:
    MhrjL = Mhrj00D2V(j,nh,uhL,vhthjL,OrnL)
"""

function Mhrj00D2V(j::Int, nh::T, vhthjL::T, OrnL::T) where {T <: Real}
  
  if j == 0
    return Mhr000D2V(nh)
  else
    return nh * vhthjL * (1 + OrnL)
  end
end

function Mhrj00D2V(j::Int, nh::T, OrnL::T) where {T <: Real}
  
  if j == 0
    return Mhr000D2V(nh)
  else
    return nh * (1 + OrnL)
  end
end

"""
  When `j == L`

  Inputs:
  Outputs:
    MhrjL = MhrLL0D2V(L,nh,uhL,vhthjL,OrnL)
"""

function MhrLL0D2V(L::Int, nh::T, uhL::T) where {T <: Real}
  
  if L == 0
    return Mhr000D2V(nh)
  else
    return nh * uhL
  end
end


function MhrLL0D2V(L::Int, uhL::T) where {T <: Real}
  
  if L == 0
    return 1.0
  else
    return uhL
  end
end

function Mhr000D2V(nh::T) where {T <: Real}
  
  return nh
end
