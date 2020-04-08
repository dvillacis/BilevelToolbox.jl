#########################
# Some utility functions
#########################

__precompile__()

module Util

##############
# Our exports
##############

export map_first_slice!,
       reduce_first_slice,
       norm₂,
       γnorm₂,
       norm₂w,
       norm₂²,
       norm₂w²,
       norm₂₁,
       γnorm₂₁,
       dot,
       mean,
       proj_norm₂₁ball!,
       curry,
       ⬿,
       @threadsif,
       @background,
       @backgroundif


##########
# Threads
##########

macro threadsif(threads, loop)
    return esc(:(if $threads
                    Threads.@threads $loop
                else
                    $loop
                end))
end

macro background(bgtask, fgtask)
    return :(t = Threads.@spawn $(esc(bgtask));
             $(esc(fgtask));
             wait(t))
end

macro backgroundif(threads, bgtask, fgtask)
    return :(if $(esc(threads))
                @background $(esc(bgtask)) $(esc(fgtask))
            else
                $(esc(bgtask))
                $(esc(fgtask))
            end)
end

########################
# Functional programming
#########################

curry = (f::Function,y...)->(z...)->f(y...,z...)

###############################
# For working with NamedTuples
###############################

⬿ = merge

######
# map
######

@inline function map_first_slice!(f!, y)
    for i in CartesianIndices(size(y)[2:end])
        @inbounds f!(@view(y[:, i]))
    end
end

@inline function map_first_slice!(x, f!, y)
    for i in CartesianIndices(size(y)[2:end])
        @inbounds f!(@view(x[:, i]), @view(y[:, i]))
    end
end

@inline function reduce_first_slice(f, y; init=0.0)
    accum=init
    for i in CartesianIndices(size(y)[2:end])
        @inbounds accum=f(accum, @view(y[:, i]))
    end
    return accum
end

###########################
# Norms and inner products
###########################

@inline function dot(x, y)
    @assert(length(x)==length(y))

    accum=0
    for i=1:length(y)
        @inbounds accum += x[i]*y[i]
    end
    return accum
end

@inline function norm₂w²(y, w)
    #Insane memory allocs
    #return @inbounds sum(i -> y[i]*y[i]*w[i], 1:length(y))
    accum=0
    for i=1:length(y)
        @inbounds accum=accum+y[i]*y[i]*w[i]
    end
    return accum
end

@inline function norm₂w(y, w)
    return √(norm₂w²(y, w))
end

@inline function norm₂²(y)
    #Insane memory allocs
    #return @inbounds sum(i -> y[i]*y[i], 1:length(y))
    accum=0
    for i=1:length(y)
        @inbounds accum=accum+y[i]*y[i]
    end
    return accum
end

@inline function norm₂(y)
    return √(norm₂²(y))
end

@inline function γnorm₂(y, γ)
    hubersq = xsq -> begin
        x=√xsq
        return if x > γ
            x-γ/2
        elseif x<-γ
            -x-γ/2
        else
            xsq/(2γ)
        end
    end

    if γ==0
        return norm₂(y)
    else
        return hubersq(norm₂²(y))
    end
end

function norm₂₁(y)
    return reduce_first_slice((s, x) -> s+norm₂(x), y)
end

function γnorm₂₁(y,γ)
    return reduce_first_slice((s, x) -> s+γnorm₂(x, γ), y)
end

function mean(v)
    return sum(v)/prod(size(v))
end

@inline function proj_norm₂₁ball!(y, α)
    α²=α*α

    if ndims(y)==3 && size(y, 1)==2
        @inbounds for i=1:size(y, 2)
            @simd for j=1:size(y, 3)
                n² = y[1,i,j]*y[1,i,j]+y[2,i,j]*y[2,i,j]
                if n²>α²
                    v = α/√n²
                    y[1, i, j] *= v
                    y[2, i, j] *= v
                end
            end
        end
    else
        y′=reshape(y, (size(y, 1), prod(size(y)[2:end])))

        @inbounds @simd for i=1:size(y′, 2)# in CartesianIndices(size(y)[2:end])
            n² = norm₂²(@view(y′[:, i]))
            if n²>α²
                y′[:, i] .*= (α/√n²)
            end
        end
    end
end

end # Module

