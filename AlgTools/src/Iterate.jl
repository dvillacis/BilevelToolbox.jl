#################################
# Tools for iterative algorithms
#################################

__precompile__()

module Iterate

using Printf

##############
# Our exports
##############

export simple_iterate

########################################################################
# Simple itertion function, calling `step()` `params.maxiter` times and 
# reporting objective value every `params.verbose_iter` iterations.
# The function `step` should take as its argument a function that itself
# takes as its argument a function that calculates the objective value
# on demand.
########################################################################

function simple_iterate(step :: Function,
                        params::NamedTuple)
    for iter=1:params.maxiter
        step() do calc_objective
            if params.verbose_iter!=0 && mod(iter, params.verbose_iter) == 0
                v, _ = calc_objective()
                @printf("%d/%d J=%f\n", iter, params.maxiter, v)
                return true
            end
        end
    end
end

function simple_iterate(step :: Function,
                        datachannel::Channel{T},
                        params::NamedTuple) where T
    for iter=1:params.maxiter
        d = take!(datachannel)
        step(d) do calc_objective
            if params.verbose_iter!=0 && mod(iter, params.verbose_iter) == 0
                v, _ = calc_objective()
                @printf("%d/%d J=%f\n", iter, params.maxiter, v)
                return true
            end
        end
    end
end

end

