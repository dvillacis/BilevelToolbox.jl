####################################################################
# Nonsmooth Trust Region Algorithm
####################################################################

__precompile__()

module AlgorithmTrustRegionNS

using Printf

using AlgTools.Util
import AlgTools.Iterate
using ImageTools.Gradient
using ImageTools.Translate

###############################################
# Types (several imported from ImageTools.Translate)
###############################################

Image = Translate.Image


############
# Algorithm
############

function solve(;iterate=AlgTools.simple_iterate,
                params::NamedTuple)

    ################################                                        
    # Extract and set up parameters
    ################################

    ######################
    # Initialise iterates
    ######################


    ####################
    # Run the algorithm
    ####################
end