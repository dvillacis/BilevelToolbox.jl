######################################
# Image subpixel accuracy translation
######################################

__precompile__()

module Translate

using AlgTools.Util: @threadsif

##########
# Exports
##########

export interpolate2d,
       interpolate2d_quadrants,
       extract_subimage!,
       translate_image!,
       DisplacementFull,
       DisplacementConstant,
       Displacement,
       Image

##################
# Types
##################

# Two different types of displacement data supported:
#  a) given in each pixel
#  b) constant in space
Image = Array{Float64,2}
DisplacementFull = Array{Float64,3}
DisplacementConstant = Array{Float64,1}
Displacement = Union{DisplacementFull,DisplacementConstant}

#############################
# Base interpolation routine
#############################

@inline function interpolate2d_quadrants(v, x, y)
    (m, n) = size(v)
    clipx = xʹ -> max(1, min(xʹ, m))
    clipy = yʹ -> max(1, min(yʹ, n))

    xfℤ = clipx(floor(Int, x))
    xcℤ = clipx(ceil(Int, x))
    yfℤ = clipy(floor(Int, y))
    ycℤ = clipy(ceil(Int, y))
       
    xf = convert(Float64, xfℤ)
    xc = convert(Float64, xcℤ)
    yf = convert(Float64, yfℤ)
    yc = convert(Float64, ycℤ)
    xm = (xf+xc)/2
    ym = (yf+yc)/2

    vff = @inbounds v[xfℤ, yfℤ]
    vfc = @inbounds v[xfℤ, ycℤ]
    vcf = @inbounds v[xcℤ, yfℤ]
    vcc = @inbounds v[xcℤ, ycℤ]
    vmm = (vff+vfc+vcf+vcc)/4

    if xfℤ==xcℤ
        if yfℤ==ycℤ
            # Completely degenerate case
            v = vmm
        else
            # Degenerate x
            v = vff+(y-yf)/(yc-yf)*(vfc-vff)
        end
    elseif yfℤ==ycℤ
        # Degenerate y
        v = vff + (x-xf)/(xc-xf)*(vcf-vff)
    elseif y-ym ≥ x-xm
        # top-left half
        if (y-ym) + (x-xm) ≥ 0
            # top quadrant
            v = vfc + (x-xf)/(xc-xf)*(vcc-vfc) + (y-yc)/(ym-yc)*(vmm-(vcc+vfc)/2)
        else
            # left quadrant
            v = vff + (y-yf)/(yc-yf)*(vfc-vff) + (x-xf)/(xm-xf)*(vmm-(vfc+vff)/2)
        end
    else
        # bottom-left half
        if (y-ym) + (x-xm) ≥ 0
            # right quadrant
            v = vcf + (y-yf)/(yc-yf)*(vcc-vcf) + (x-xc)/(xm-xc)*(vmm-(vcc+vcf)/2)
        else
            # bottom quadrant
            v = vff + (x-xf)/(xc-xf)*(vcf-vff) + (y-yf)/(ym-yf)*(vmm-(vcf+vff)/2)
        end
    end

    return v
end

##############
# Translation
##############

function translate_image!(x, z, u::DisplacementFull;
                          threads::Bool=false)
    @assert(size(u, 1)==2 && size(x)==size(u)[2:end] && size(x)==size(z))

    @threadsif threads for i=1:size(x, 1)
        for j=1:size(x, 2)
            x[i, j] = interpolate2d_quadrants(z, i - u[1, i, j], j - u[2, i, j])
        end
    end
end

function translate_image!(x, z, u::DisplacementConstant;
                          threads::Bool=false)
    @assert(size(u)==(2,) && size(x)==size(z))

    @inbounds a, b = u[1], u[2]

    @threadsif threads for i=1:size(x, 1)
        for j=1:size(x, 2)
            x[i, j] = interpolate2d_quadrants(z, i - a, j - b)
        end
    end
end

######################
# Subimage extraction
######################

function extract_subimage!(b, im, v::DisplacementConstant;
                           threads::Bool=false)
    (imx, imy) = size(im)
    (bx, by) = size(b)

    # Translation from target to source coordinates
    vxʹ = (imx-bx)/2 - v[1]
    vyʹ = (imy-by)/2 - v[2]

    # Target image indices within source image
    px = min(max(ceil(Int, max(1, vxʹ + 1) - vxʹ), 1), bx)
    py = min(max(ceil(Int, max(1, vyʹ + 1) - vyʹ), 1), by)
    qx = max(min(floor(Int, min(imx, vxʹ + bx) - vxʹ), bx), 1)
    qy = max(min(floor(Int, min(imy, vyʹ + by) - vyʹ), by), 1)
    
    @inbounds begin
        b[1:px-1, :] .= 0
        b[qx+1:bx, :] .= 0
    end

    @threadsif false for i=px:qx
        @inbounds begin
            b[i, 1:py-1] .= 0
            b[i, qy+1:by] .= 0
            for j=py:qy
                b[i, j] = interpolate2d_quadrants(im, i+vxʹ, j+vyʹ)
            end
        end
    end
end

#####################################################
# Precompilation hints to speed up compilation time
# for projects depending on this package (hopefully).
######################################################

precompile(translate_image!, (Array{Float64,2}, Array{Float64,2}, Array{Float64,1}))
precompile(translate_image!, (Array{Float64,2}, Array{Float64,2}, Array{Float64,3}))
precompile(extract_subimage!, (Array{Float64,2}, Array{Float64,2}, Array{Float64,1}))

end
