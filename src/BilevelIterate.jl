#################################
# Tools for bilevel iterative algorithms
#################################

__precompile__()

module BilevelIterate

using Printf
using Setfield
using FileIO
using ColorTypes: Gray

using AlgTools.Util
using AlgTools.StructTools
using AlgTools.LinkedLists

using ImageTools.Visualise

##################
# Helper routines
##################

@inline function secs_ns()
    return convert(Float64, time_ns()) * 1e-9
end

clip = x->min(max(x, 0.0), 1.0)
grayimg = im->Gray.(clip.(im))

##################
# Data structures
##################

struct LogEntry <: IterableStruct
    iter::Int
    time::Float64
    function_value::Float64
    λ::Float64
    radius::Float64
    grad::Float64
    B::Float64
    ρ::Float64
end

struct BilevelState
    vis::Union{Channel,Bool,Nothing}
    visproc::Union{Nothing,Task}
    start_time::Union{Real,Nothing}
    wasted_time::Real
    log::LinkedList{LogEntry}
end

##############
# Our exports
##############

export bilevel_iterate,
        BilevelState

function bilevel_iterate(   st::BilevelState,
                            step::Function,
                            params::NamedTuple)
    try
        for iter = 1:params.maxiter
            st = step() do calc_objective
                if isnothing(st.start_time)
                    st = @set st.start_time = secs_ns()
                end

                if params.verbose_iter != 0 && mod(iter, params.verbose_iter) == 0
                    verb_start = secs_ns()
                    tm = verb_start - st.start_time - st.wasted_time
                    value, λ, radius, grad, B, ρ, x = calc_objective()

                    entry = LogEntry(iter, tm, value, λ, radius, grad, B, ρ)
                    st = @set st.log = LinkedListEntry(entry, st.log)

                    @printf("%d/%d J=%f λ=%f, radius=%f, grad=%f, B=%f, ρ=%f\n", iter, params.maxiter, value, λ, radius, grad, B, ρ)
                    visualise(st.vis, (grayimg(x),))
                    
                    if params.save_iterations
                        fn = t->"$(params.save_prefix)_$(t)_iter$(iter).png"
                        save(File(format"PNG", fn("reco")), grayimg(x))
                    end
                end
                return st
            end
        end
    catch ex
        if isa(ex, InterruptException)
            printstyled("\rUser interrupt—finishing up.\n", bold = true, color = 202)
        else
            throw(ex)
        end
    end
    return st
end

end # Module