
"""

  Inputs:
    p = [Ncons; Mhst]
  Outputs:
    consMhjL!(res,xx,p)
"""

function consMhjL!(res,xx,p)

    # Ncons = p[1]
    Mhst = p[2:end]
    if p[1] == 1
        res .= [sum(xx[1:3:end] .* xx[2:3:end].^L) - Mhst[1]]
    elseif p[1] == 2
        res .= [sum(xx[1:3:end] .* xx[2:3:end].^L) - Mhst[1],
                sum((xx[1:3:end] .* (xx[3:3:end]).^2 .* xx[2:3:end].^L) .* (1 .+ CjLk(2,L,1) * (xx[2:3:end]).^2 ./ (xx[3:3:end]).^2)) - Mhst[2]]
    elseif p[1] == 3
        res .= [sum(xx[1:3:end] .* xx[2:3:end].^L) - Mhst[1],
               sum((xx[1:3:end] .* (xx[3:3:end]).^2 .* xx[2:3:end].^L) .* (1 .+ CjLk(2,L,1) * (xx[2:3:end]).^2 ./ (xx[3:3:end]).^2)) - Mhst[2],
               sum((xx[1:3:end] .* (xx[3:3:end]).^2 .* xx[2:3:end].^L) .* (1 .+ CjLk(4,L,1) * (xx[2:3:end]).^2 ./ (xx[3:3:end]).^2 .+ CjLk(4,L,2) * (xx[2:3:end]).^4 ./ (xx[3:3:end]).^4)) - Mhst[3]]
    else
        dgtfnhggnn
    end
    return nothing
end



# function consMhjL!(res::AbstractVector{T},xx::AbstractVector{T},p::AbstractVector{T}) where{T <:Real}

#     # Ncons = p[1]
#     Mhst = p[2:end]
#     if p[1] == 1
#         res .= [sum(xx[1:3:end] .* xx[2:3:end].^L) - Mhst[1]]
#     elseif p[1] == 2
#         res .= [sum(xx[1:3:end] .* xx[2:3:end].^L) - Mhst[1],
#                 sum((xx[1:3:end] .* (xx[3:3:end]).^2 .* xx[2:3:end].^L) .* (1 .+ CjLk(2,L,1) * (xx[2:3:end]).^2 ./ (xx[3:3:end]).^2)) - Mhst[2]]
#     elseif p[1] == 3
#         res .= [sum(xx[1:3:end] .* xx[2:3:end].^L) - Mhst[1],
#                sum((xx[1:3:end] .* (xx[3:3:end]).^2 .* xx[2:3:end].^L) .* (1 .+ CjLk(2,L,1) * (xx[2:3:end]).^2 ./ (xx[3:3:end]).^2)) - Mhst[2],
#                sum((xx[1:3:end] .* (xx[3:3:end]).^2 .* xx[2:3:end].^L) .* (1 .+ CjLk(4,L,1) * (xx[2:3:end]).^2 ./ (xx[3:3:end]).^2 .+ CjLk(4,L,2) * (xx[2:3:end]).^4 ./ (xx[3:3:end]).^4)) - Mhst[3]]
#     else
#         dgtfnhggnn
#     end
#     return nothing
# end


# # For `ReverseDiff.jl`
# function consMhjL!(res::AbstractVector{T},xx::TA,p::AbstractVector{Tf}) where{T <:Real,Tf <:Real,TA}

#     # Ncons = p[1]
#     Mhst = p[2:end]
#     if p[1] == 1
#         res .= [sum(xx[1:3:end] .* xx[2:3:end].^L) - Mhst[1]]
#     elseif p[1] == 2
#         res .= [sum(xx[1:3:end] .* xx[2:3:end].^L) - Mhst[1],
#                 sum((xx[1:3:end] .* (xx[3:3:end]).^2 .* xx[2:3:end].^L) .* (1 .+ CjLk(2,L,1) * (xx[2:3:end]).^2 ./ (xx[3:3:end]).^2)) - Mhst[2]]
#     elseif p[1] == 3
#         res .= [sum(xx[1:3:end] .* xx[2:3:end].^L) - Mhst[1],
#                sum((xx[1:3:end] .* (xx[3:3:end]).^2 .* xx[2:3:end].^L) .* (1 .+ CjLk(2,L,1) * (xx[2:3:end]).^2 ./ (xx[3:3:end]).^2)) - Mhst[2],
#                sum((xx[1:3:end] .* (xx[3:3:end]).^2 .* xx[2:3:end].^L) .* (1 .+ CjLk(4,L,1) * (xx[2:3:end]).^2 ./ (xx[3:3:end]).^2 .+ CjLk(4,L,2) * (xx[2:3:end]).^4 ./ (xx[3:3:end]).^4)) - Mhst[3]]
#     else
#         dgtfnhggnn
#     end
#     return nothing
# end

# # For `AutoZygote.jl`
# function consMhjL!(res,xx::AbstractVector{T},p::AbstractVector{T}) where{T <:Real}

#     # Ncons = p[1]
#     Mhst = p[2:end]
#     if p[1] == 1
#         res .= [sum(xx[1:3:end] .* xx[2:3:end].^L) - Mhst[1]]
#     elseif p[1] == 2
#         res .= [sum(xx[1:3:end] .* xx[2:3:end].^L) - Mhst[1],
#                 sum((xx[1:3:end] .* (xx[3:3:end]).^2 .* xx[2:3:end].^L) .* (1 .+ CjLk(2,L,1) * (xx[2:3:end]).^2 ./ (xx[3:3:end]).^2)) - Mhst[2]]
#     elseif p[1] == 3
#         res .= [sum(xx[1:3:end] .* xx[2:3:end].^L) - Mhst[1],
#                sum((xx[1:3:end] .* (xx[3:3:end]).^2 .* xx[2:3:end].^L) .* (1 .+ CjLk(2,L,1) * (xx[2:3:end]).^2 ./ (xx[3:3:end]).^2)) - Mhst[2],
#                sum((xx[1:3:end] .* (xx[3:3:end]).^2 .* xx[2:3:end].^L) .* (1 .+ CjLk(4,L,1) * (xx[2:3:end]).^2 ./ (xx[3:3:end]).^2 .+ CjLk(4,L,2) * (xx[2:3:end]).^4 ./ (xx[3:3:end]).^4)) - Mhst[3]]
#     else
#         dgtfnhggnn
#     end
#     return nothing
# end
