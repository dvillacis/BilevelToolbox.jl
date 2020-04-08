########################
# Discretised gradients
########################

__precompile__()

module Gradient

##############
# Our exports
##############

export ∇₂!, ∇₂ᵀ!, ∇₂fold!,
       ∇₂_norm₂₂_est, ∇₂_norm₂₂_est²,
       ∇₂_norm₂∞_est, ∇₂_norm₂∞_est²,
       ∇₂c!, ∇₂cfold!,
       ∇₃!, ∇₃ᵀ!,
       vec∇₃!, vec∇₃ᵀ!

##################
# Helper routines
##################

@inline function imfold₂′!(f_aa!, f_a0!, f_ab!,
                           f_0a!, f_00!, f_0b!,
                           f_ba!, f_b0!, f_bb!,
                           n, m, state)
    # First row
    state = f_aa!(state, (1, 1))
    for j = 2:m-1
        state = f_a0!(state, (1, j))
    end
    state = f_ab!(state, (1, m))

    # Middle rows
    for i=2:n-1
        state = f_0a!(state, (i, 1))
        for j = 2:m-1
            state = f_00!(state, (i, j))
        end
        state = f_0b!(state, (i, m))
    end

    # Last row
    state = f_ba!(state, (n, 1))
    for  j =2:m-1
        state = f_b0!(state, (n, j))
    end
    return f_bb!(state, (n, m))
end

#########################
# 2D forward differences
#########################

∇₂_norm₂₂_est² = 8
∇₂_norm₂₂_est = √∇₂_norm₂₂_est²
∇₂_norm₂∞_est² = 2
∇₂_norm₂∞_est = √∇₂_norm₂∞_est²

function ∇₂!(u₁, u₂, u)
    @. @views begin
        u₁[1:(end-1), :] = u[2:end, :] - u[1:(end-1), :]
        u₁[end, :, :] = 0

        u₂[:, 1:(end-1)] = u[:, 2:end] - u[:, 1:(end-1)]
        u₂[:, end] = 0
    end
    return u₁, u₂
end

function ∇₂!(v, u)
    ∇₂!(@view(v[1, :, :]), @view(v[2, :, :]), u)
end

@inline function ∇₂fold!(f!::Function, u, state)
    @inline function g!(state, pt)
        (i, j) = pt
        g = @inbounds [u[i+1, j]-u[i, j], u[i, j+1]-u[i, j]]
        return f!(g, state, pt)
    end
    @inline function gr!(state, pt)
        (i, j) = pt
        g = @inbounds [u[i+1, j]-u[i, j], 0.0]
        return f!(g, state, pt)
    end
    @inline function gb!(state, pt)
        (i, j) = pt
        g = @inbounds [0.0, u[i, j+1]-u[i, j]]
        return f!(g, state, pt)
    end
    @inline function g0!(state, pt)
        return f!([0.0, 0.0], state, pt)
    end
    return imfold₂′!(g!, g!, gr!,
                     g!, g!, gr!,
                     gb!, gb!, g0!,
                     size(u, 1), size(u, 2), state)
end

function ∇₂ᵀ!(v, v₁, v₂)
    @. @views begin
        v[2:(end-1), :] = v₁[1:(end-2), :] - v₁[2:(end-1), :]
        v[1, :] = -v₁[1, :]
        v[end, :] = v₁[end-1, :]

        v[:, 2:(end-1)] += v₂[:, 1:(end-2)] - v₂[:, 2:(end-1)]
        v[:, 1] += -v₂[:, 1]
        v[:, end] += v₂[:, end-1]
    end
    return v
end

function ∇₂ᵀ!(u, v)
    ∇₂ᵀ!(u, @view(v[1, :, :]), @view(v[2, :, :]))
end

##################################################
# 2D central differences (partial implementation)
##################################################

function ∇₂c!(v, u)
    @. @views begin
        v[1, 2:(end-1), :] = (u[3:end, :] - u[1:(end-2), :])/2
        v[1, end, :] = (u[end, :] - u[end-1, :])/2
        v[1, 1, :] = (u[2, :] - u[1, :])/2

        v[2, :, 2:(end-1)] = (u[:, 3:end] - u[:, 1:(end-2)])/2
        v[2, :, end] = (u[:, end] - u[:, end-1])/2
        v[2, :, 1] = (u[:, 2] - u[:, 1])/2
    end
end

@inline function ∇₂cfold!(f!::Function, u, state)
    n, m = size(u)
    @inline function g!(state, pt)
        (i, j) = pt
        g = @inbounds [(u[i+1, j]-u[i-1, j])/2, (u[i, j+1]-u[i, j-1])/2]
        return f!(g, state, pt)
    end
    @inline function gb!(state, pt)
        (i, j) = pt
        g = @inbounds [(u[min(i+1,n), j]-u[max(i-1,1), j])/2,
                       (u[i, min(j+1,m)]-u[i, max(j-1,1)])/2]
        return f!(g, state, pt)
    end
    return imfold₂′!(gb!, gb!, gb!,
                     gb!, g!,  gb!,
                     gb!, gb!, gb!,
                     size(u, 1), size(u, 2), state)
end

#########################
# 3D forward differences
#########################

function ∇₃!(u₁,u₂,u₃,u)
    @. @views begin
        u₁[1:(end-1), :, :] = u[2:end, :, :] - u[1:(end-1), :, :]
        u₁[end, :, :] = 0

        u₂[:, 1:(end-1), :] = u[:, 2:end, :] - u[:, 1:(end-1), :]
        u₂[:, end, :] = 0

        u₃[:, :, 1:(end-1)] = u[:, :, 2:end] - u[:, :, 1:(end-1)]
        u₃[:, :, end] = 0
    end
    return u₁, u₂, u₃
end

function ∇₃ᵀ!(v,v₁,v₂,v₃)
    @. @views begin
        v[2:(end-1), :, :] = v₁[1:(end-2), :, :] - v₁[2:(end-1), :, :]
        v[1, :, :] = -v₁[1, :, :]
        v[end, :, :] = v₁[end-1, :, :]

        v[:, 2:(end-1), :] += v₂[:, 1:(end-2), :] - v₂[:, 2:(end-1), :]
        v[:, 1, :] += -v₂[:, 1, :]
        v[:, end, :] += v₂[:, end-1, :]

        v[:, :, 2:(end-1)] += v₃[:, :, 1:(end-2)] - v₃[:, :, 2:(end-1)]
        v[:, :, 1] += -v₃[:, :, 1]
        v[:, :, end] += v₃[:, :, end-1]
    end
    return v
end

###########################################
# 3D forward differences for vector fields
###########################################

function vec∇₃!(u₁,u₂,u₃,u)
    @. @views for j=1:size(u, 1)
        ∇₃!(u₁[j, :, :, :],u₂[j, :, :, :],u₃[j, :, :, :],u[j, :, :, :])
    end
    return u₁, u₂, u₃
end

function vec∇₃ᵀ!(u,v₁,v₂,v₃)
    @. @views for j=1:size(u, 1)
        ∇₃ᵀ!(u[j, :, :, :],v₁[j, :, :, :],v₂[j, :, :, :],v₃[j, :, :, :])
    end
    return u
end

#####################################################
# Precompilation hints to speed up compilation time
# for projects depending on this package (hopefully).
######################################################

precompile(∇₂!, (Array{Float64,2}, Array{Float64,2}, Array{Float64,2}))
precompile(∇₂!, (Array{Float64,3}, Array{Float64,2}))
precompile(∇₂ᵀ!, (Array{Float64,2}, Array{Float64,2}, Array{Float64,2}))
precompile(∇₂ᵀ!, (Array{Float64,2}, Array{Float64,3}))
precompile(∇₂c!, (Array{Float64,3}, Array{Float64,2}))
precompile(∇₃!, (Array{Float64,3}, Array{Float64,3}, Array{Float64,3},Array{Float64,3}))
precompile(∇₃ᵀ!, (Array{Float64,3}, Array{Float64,3}, Array{Float64,3},Array{Float64,3}))
precompile(vec∇₃!, (Array{Float64,4}, Array{Float64,4}, Array{Float64,4},Array{Float64,4}))
precompile(vec∇₃ᵀ!, (Array{Float64,4}, Array{Float64,4}, Array{Float64,4},Array{Float64,4}))

# The folding functions cannot be precompiled as theyre' meant to be (hopefully)
# inlined in such a way that the parameter function also gets inlined withou our
# code

end # Module
