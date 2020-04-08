#################################
# Tools for working with structs
#################################

__precompile__()

module StructTools

##############
# Our exports
##############

export replace,
       IterableStruct

######################################################
# Replace entries by those given as keyword arguments
######################################################

function replace(base::T; kw...) where T
    k = keys(kw)
    T([n ∈ k ? kw[n] : getfield(base, n) for n ∈ fieldnames(T)]...)
end

#########################################################
# Iteration of structs.
# One only needs to make them instance of IterableStruct
#########################################################

abstract type IterableStruct end

function Base.iterate(s::T) where T <: IterableStruct
    return Base.iterate(s, (0, fieldnames(T)))
end

function Base.iterate(
    s::T, st::Tuple{Integer,NTuple{N,Symbol}}
) where T <: IterableStruct where N
    (i, k)=st
    return (i<N ? (getfield(s, i+1), (i+1, k)) : nothing)
end

end
