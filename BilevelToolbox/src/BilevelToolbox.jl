##################
# Our main module
##################

##################
# Module for generating different types of noise contaminated images
##################

__precompile__()

module BilevelToolbox

########################
# Load external modules
########################

using GR

using AlgTools.Util
using AlgTools.StructTools
using AlgTools.LinkedLists
using AlgTools.Comms

using ImageTools.Visualise: grayimg, visualise, clip

#####################
# Load local modules
#####################

include("ImGenerate.jl")

using .ImGenerate

##############
# Our exports
##############

export show_images

###################################
# Parameterisation and experiments
###################################

struct Experiment
    mod :: Module
    imgen :: ImGen
    params :: NamedTuple
end

const default_save_prefix="img/"

const default_params = (
    noise_level = 0.1,
    α = 0.2
)

################
# Log
################

struct LogEntry <: IterableStruct
    iter :: Int
    time :: Float64
    function_value :: Float64
    psnr :: Float64
    ssim :: Float64
    psnr_data :: Float64
    ssim_data :: Float64
end

###############
# Main routine
###############

function show_images()
    imgen = generate_noisy_image("lighthouse",default_params)
    sz = size(imgen.im_true)
    
    GR.setwindow(0,sz[1],0,sz[2])
    GR.drawimage(0,1,0,1,sz[1],sz[2],grayimg(imgen.im_true)')
    #GR.drawimage(0,2,0,1,sz[1],sz[2],grayimg(imgen.im_noisy)')
end

end
