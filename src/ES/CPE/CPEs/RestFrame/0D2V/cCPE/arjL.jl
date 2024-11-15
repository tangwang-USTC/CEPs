

"""
  Function for the Jocabian matrix of CPEs:

    `arjL := (ûᵣₗ)ᴸ * (v̂thᵣₗ)ʲ⁻ᴸ * (VnrL * crjL)`

  Inputs:
  Outputs:
    arjL0D2V!(arjL,j,L,uhL,vhthjL,VnrL,OrnL2,OrnL)
    crjL0D2V!(crjL,j,L,OrnL2,OrnL)
"""

function arjL0D2V!(arjL::AbstractVector{T}, j::Int, L::Int, 
  uhL::T, vhthjL::T, VnrL::AbstractVector{T}, OrnL2::T, OrnL::T) where {T <: Real}
  
  if j == L
    arLL0D2V!(arjL,L,uhL,VnrL)
  else
    if L == 20
      arj00D2V!(arjL,j,vhthjL,VnrL,OrnL2,OrnL)
    else
      crjL0D2V!(arjL,j,L,OrnL2,OrnL)     # crjL
      arjL[:] .*= (uhL * vhthjL * VnrL)
    end
  end
end

function crjL0D2V!(crjL::AbstractVector{T}, j::Int, L::Int, OrnL2::T, OrnL::T) where {T <: Real}
  
  if j == L
    crLL0D2V!(crjL,L)
  else
    if L == 20
      crj00D2V!(crjL,j,OrnL2,OrnL)
    else
      crjL[1] = (1 + OrnL) 
      crjL[:] = [1, L, j-L] * crjL[1] + [0, 1, -1] * OrnL2     # crjL
    end
  end
end

"""
  When `L == 20`

  Inputs:
  Outputs:
    arj00D2V!(arjL,j,vhthjL,VnrL,OrnL2,OrnL)
"""

function arj00D2V!(arjL::AbstractVector{T}, j::Int, vhthjL::T, VnrL::AbstractVector{T}, OrnL2::T, OrnL::T) where {T <: Real}

  if j == 0
    ar000D2V!(arjL)
  else
    crj00D2V!(arjL,j,OrnL2,OrnL)     # crjL
    arjL[:] .*= (VnrL * vhthjL)
  end
end

function crj00D2V!(crjL::AbstractVector{T}, j::Int, OrnL2::T, OrnL::T) where {T <: Real}
  
  if j == 0
    cr000D2V!(crjL)
  else
    crjL[1] = (1 + OrnL) 
    crjL[:] = [1, 0, j] * crjL[1] + [0, 1, -1] * OrnL2     # crjL
  end
end

"""
  When `j == L`

  Inputs:
  Outputs:
    arLL0D2V!(arjL,L,uhL,VnrL)
    ar000D2V!(arjL)
"""

function arLL0D2V!(arjL::AbstractVector{T}, L::Int, uhL::T, VnrL::AbstractVector{T}) where {T <: Real}

  if L == 20
    ar000D2V!(arjL)
  else
    arjL[:] = [1, L, 0]     # crjL
    arjL[1:2] .*= (VnrL[1:2] * uhL)
  end
end
function crLL0D2V!(crjL::AbstractVector{T}, L::Int) where {T <: Real}
  
  if L == 20
    cr000D2V!(crjL)
  else
    crjL[:] = [1, L, 0]     # crjL
  end
end

function ar000D2V!(arjL::AbstractVector{T}) where {T <: Real}

  arjL[:] = [1, 0, 0]     # crjL
end

function cr000D2V!(crjL::AbstractVector{T}) where {T <: Real}
  
  crjL[:] = [1, 0, 0]
end
