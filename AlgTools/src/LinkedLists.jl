####################################################################
# Immutable linked lists (different from the mutable lists of
# https://github.com/ChrisRackauckas/LinkedLists.jl)
####################################################################

__precompile__()

module LinkedLists

using DelimitedFiles

using ..StructTools

##############
# Our exports
##############

export LinkedListEntry,
       LinkedList,
       unfold_linked_list,
       write_log

#############
# Data types
#############

struct LinkedListEntry{T}
    value :: T
    next :: Union{LinkedListEntry{T},Nothing}
end

LinkedList{T} = Union{LinkedListEntry{T},Nothing}

############
# Functions
############

function Base.iterate(list::LinkedList{T}) where T
    return Base.iterate(list, list)
end

function Base.iterate(list::LinkedList{T}, tail::Nothing) where T
    return nothing
end

function Base.iterate(list::LinkedList{T}, tail::LinkedListEntry{T}) where T
    return tail.value, tail.next
end

# Return the items in the list with the tail first
function unfold_linked_list(list::LinkedList{T}) where T
    res = []
    for value ∈ list
        push!(res, value)
    end
    return reverse(res)
end

# Write out a a “log” of LinkedList of IterableStructs as a delimited file
function write_log(filename::String, log::LinkedList{T}, comment::String) where T <: IterableStruct
    open(filename, "w") do io
        print(io, comment)
        writedlm(io, [String.(fieldnames(T))])
        writedlm(io, unfold_linked_list(log))
    end
end

end
