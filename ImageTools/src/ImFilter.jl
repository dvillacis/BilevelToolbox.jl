##################################################
# Simple (and fast for small filters compared to
# ImageFiltering) image filtering
##################################################

__precompile__()

module ImFilter

using OffsetArrays
using AlgTools.Util: @threadsif

##########
# Exports
##########

export simple_imfilter,
       gaussian

##############
# The routine
##############

@inline function inside(i, aʹ, bʹ, a, b)
     return (max(a, i - aʹ) - i):(min(b,  i + bʹ) - i)
end

function simple_imfilter(b::Array{Float64,2},
                         kernel::OffsetArray{Float64,2,Array{Float64,2}};
                         threads::Bool=true)

    n, m = size(b)
    k, 𝓁 = size(kernel)
    o₁, o₂ = kernel.offsets
    a₁, a₂ = k + o₁, 𝓁 + o₂
    b₁, b₂ = -1 - o₁, -1 - o₂
    kp = kernel.parent

    @assert(isodd(k) && isodd(𝓁))

    res = similar(b)

    @threadsif threads for i=1:n
        @inbounds for j=1:m
            tmp = 0.0
            it₁ = inside(i, a₁, b₁, 1, n)
            it₂ = inside(j, a₂, b₂, 1, m)
            for p=it₁
                @simd for q=it₂
                    tmp += kp[p-o₁, q-o₂]*b[i+p,j+q]
                end
            end
            res[i, j] = tmp
        end
    end

    return res
end

######################################################
# Distributions. Just to avoid the long load times of
# ImageFiltering and heavy dependencies on FFTW etc.
######################################################

function gaussian(σ, n)
    @assert(all(isodd.(n)))
    a=convert.(Integer, @. (n-1)/2)
    g=OffsetArray{Float64}(undef, [-m:m for m in a]...);
    for i in CartesianIndices(g)
        g[i]=exp(-sum(Tuple(i).^2 ./ (2 .* σ.^2)))
    end
    g./=sum(g)
end

end # Module

