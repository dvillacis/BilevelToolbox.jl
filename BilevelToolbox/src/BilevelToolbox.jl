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
include("AlgorithmTrustRegionNS.jl")

import .AlgorithmTrustRegionNS

using .DatasetGen

##############
# Our exports
##############

export run_bilevel_algorithm

###################################
# Parameterisation and experiments
###################################

struct Experiment
    mod :: Module
    dataset :: Dataset
    params :: NamedTuple
end

function Base.show(io::IO, e::Experiment)
    print(io, "
    mod: $(e.mod)
    dataset: $(e.dataset.name)
    params: $(e.params)
    ")
end

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

function run_bilevel_algorithm(;visualise=true, 
                                experiments,
                                kwargs...)
    for e ∈ experiments
        x,y,st = e.mod.solve()
    end
    
end

end
