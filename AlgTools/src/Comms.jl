#########################################
# Helpers for communication via channels
#########################################

module Comms

__precompile__()

##############
# Our exports
##############

export process_channel,
       put_onlylatest!,
       put_unless_closed!

####################
# Channel iteration
####################

function process_channel(fn, rc)
    while true
        d=take!(rc)
        # Take only the latest image to visualise
        while isready(rc)
            d=take!(rc)
        end
        # We're done if we were fed nothing
        if isnothing(d)
            break
        end
        try
            fn(d)
        catch ex
            error("Exception in process_channel handler. Terminating.\n")
            rethrow(ex)
        end 
    end
end

#############################################
# Ensure only latest data is in a Channel(1)
#############################################

function put_onlylatest!(rc, d)
    while isready(rc)
        take!(rc)
    end
    put!(rc, d)
end

############################################
# Cracefully return false if channel closed
############################################

function put_unless_closed!(rc, d)
    try
        put!(rc, d)
    catch ex
        if isa(ex, InvalidStateException) && ex.state==:closed
            return false
        else
            rethrow(ex)
        end
    end
    return true
end

end # Module
