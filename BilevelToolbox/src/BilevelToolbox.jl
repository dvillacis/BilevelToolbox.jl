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

using GR, Random

using AlgTools.Util
using AlgTools.StructTools
using AlgTools.LinkedLists
using AlgTools.Comms

using ImageTools.Visualise: grayimg, visualise, clip

#####################
# Load local modules
#####################

include("DatasetGen.jl")
include("Denoise.jl")

using .DatasetGen

##############
# Our exports
##############

export show_images

###################################
# Parameterisation and experiments
###################################

struct Experiment
    mod :: Module
    dataset :: Dataset
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

    srand(seed) = Random.seed!(seed)
    srand(37)
    y = randn(20, 500)

    GR.setviewport(0.1, 0.95, 0.1, 0.95)
    GR.setcharheight(0.020)
    GR.settextcolorind(82)
    GR.setfillcolorind(90)
    GR.setfillintstyle(1)

    for x in 1:5000
        GR.clearws()
        GR.setwindow(x, x+500, -200, 200)
        GR.fillrect(x, x+500, -200, 200)
        GR.setlinecolorind(0);  GR.grid(50, 50, 0, -200, 2, 2)
        GR.setlinecolorind(82); GR.axes(50, 50, x, -200, 2, 2, -0.005)
        y = hcat(y, randn(20))
        for i in 1:20
            GR.setlinecolorind(980 + i)
            s = cumsum(reshape(y[i,:], x+500))
            GR.polyline([x:x+500;], s[x:x+500])
        end
        GR.updatews()
    end
end

end
