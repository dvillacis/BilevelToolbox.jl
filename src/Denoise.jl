########################################################
# Basic TV denoising via primal–dual proximal splitting
########################################################

__precompile__()

module Denoise

using AlgTools.Util
import AlgTools.Iterate
using ImageTools.Gradient

##############
# Our exports
##############

export  denoise_rof_pdhg,
        denoise_tvl1_pdhg

#############
# Data types
#############

ImageSize = Tuple{Integer,Integer}
Image = Array{Float64,2}
Primal = Image
Dual = Array{Float64,3}

#########################
# Iterate initialisation
#########################

function init_rest(x::Primal)
    imdim=size(x)

    y = zeros(2, imdim...)
    Δx = copy(x)
    Δy = copy(y)
    x̄ = copy(x)

    return x, y, Δx, Δy, x̄
end

function init_primal(xinit::Image, b)
    return copy(xinit)
end

function init_primal(xinit::Nothing, b :: Image)
    return zeros(size(b)...)
end


############
# Algorithm
############

"""
    denoise_rof_pdhg(b; xinit,iterate,params)

Denoise a given image noisy image b using the primal–dual hybrid gradient method
"""
function denoise_rof_pdhg(  b :: Image;
                            xinit :: Union{Image,Nothing} = nothing,
                            iterate = AlgTools.simple_iterate,
                            params :: NamedTuple)

    ################################                                        
    # Extract and set up parameters
    ################################                    

    λ, τ = params.λ, params.τ     
    σ = 1/τ/∇₂_norm₂₂_est²

    ######################
    # Initialise iterates
    ######################

    x, y, Δx, Δy, x̄ = init_rest(init_primal(xinit, b))

    ####################
    # Run the algorithm
    ####################

    v = iterate(params) do verbose :: Function
        ∇₂ᵀ!(Δx, y)                                 # primal step:
        @. x̄ = x                                    # |  save old x for over-relax
        @. x = (x-τ*(Δx-params.λ*b))/(1+params.λ*τ) # |  prox
        @. x̄ = 2*x-x̄                                # | over-relaxation
        ∇₂!(Δy, x̄)                                  # dual step: y
        @. y = (y+σ*Δy)
        proj_norm₂₁ball!(y, 1)                      # |  prox
    

        ################################
        # Give the primal dual gap value if needed
        ################################
        v = verbose() do
            ∇₂!(Δy, x)
            ∇₂ᵀ!(Δx, y) 
            primal = params.λ*norm₂²(x-b)/2 + norm₂₁(Δy)
            dual = 0.5*params.λ*norm₂²((1/params.λ)*Δx+b)-0.5*params.λ*norm₂²(b)#norm₂²(Δx+b)/2
            value = dual#primal-dual
            value,x,y
        end
        v
    end
    return x,y,v
end

function denoise_tvl1_pdhg( b :: Image;
                            xinit :: Union{Image,Nothing} = nothing,
                            iterate = AlgTools.simple_iterate,
                            params :: NamedTuple)
end

end