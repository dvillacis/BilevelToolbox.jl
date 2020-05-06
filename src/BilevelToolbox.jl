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
using FileIO

using AlgTools.Util
using AlgTools.StructTools
using AlgTools.LinkedLists
using AlgTools.Comms

using ImageTools.Visualise

#####################
# Load local modules
#####################

include("DatasetGen.jl")
include("Denoise.jl")
include("BilevelIterate.jl")
include("AlgorithmTrustRegionNS.jl")

import .AlgorithmTrustRegionNS

using .DatasetGen
using .BilevelIterate

##############
# Our exports
##############

export  Experiment,
        run_bilevel_algorithm

###################################
# Parameterisation and experiments
###################################

struct Experiment
    mod :: Module
    dataset :: AbstractDataset
    lower_level_solver :: Function
    upper_level_cost :: Function
    upper_level_gradient :: Function
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

struct LogEntryHiFi <: IterableStruct
    iter :: Int
    v_cumul_true_y :: Float64
    v_cumul_true_x :: Float64
end

####################
# Launcher routines
####################

function initialise_bilevel_visualisation(visualise; iterator=bilevel_iterate)
    # Create visualisation
    if visualise
        rc = Channel(1)
        visproc = Threads.@spawn bg_visualise(rc)
        bind(rc, visproc)
        vis = rc
    else
        vis = false
        visproc = nothing
    end

    st = BilevelState(vis, visproc, nothing, 0.0, nothing)
    iterate = curry(bilevel_iterate, st)

    return st, iterate
end

###############
# Main routine
###############

function run_bilevel_algorithm(experiment :: Experiment; visualise=true)

    println("Running Bilevel Algorithm: $experiment")

    st,iterate = initialise_bilevel_visualisation(true)
    λ,x,st = experiment.mod.solve(experiment.dataset, experiment.lower_level_solver, experiment.upper_level_cost, experiment.upper_level_gradient; iterate = iterate, params=experiment.params)
    finalise_visualisation(st)
    if experiment.params.save_results
        im_true,im_noisy = get_training_pair(1,experiment.dataset)
        save_prefix = "bilevel_run_" * experiment.dataset.name
        perffile = save_prefix * ".txt"
        println("Saving " * perffile)
        write_log(perffile, st.log, "# params = $(experiment.params)\n")
        fn = (t, ext) -> "$(save_prefix)_$(t).$(ext)"
        save(File(format"PNG", fn("true", "png")), grayimg(im_true))
        save(File(format"PNG", fn("data", "png")), grayimg(im_noisy))
        save(File(format"PNG", fn("reco", "png")), grayimg(x))
    end
    println("Wasted time: $(st.wasted_time)s")
    
end

end
