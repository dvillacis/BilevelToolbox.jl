###################
# Image generation
###################

module DatasetGen

using ColorTypes: Gray
import TestImages
import QuartzImageIO

using AlgTools.Util

using ImageTools.Translate

###############################################
# Types (several imported from ImageTools.Translate)
###############################################

Image = Translate.Image

##############
# Our exports
##############

export  Dataset,
        Entry,
        generate_dataset,
        generate_synthetic_noise_entry

##################
# Data structures
##################

struct Entry
    im_true :: Image
    im_noisy :: Image
    name :: String
    dim :: Tuple{Int64,Int64}
    dynrange :: Float64
end

struct Dataset
    name :: String
    entries :: Array{Entry,1}
end

Base.show(io::IO, ::Type{Entry}) = print(io,"this is an entry")

##################
# Synthetic noise single dataset
##################

"""
    generate_synthetic_noise_entry(imname,params)

Generate a single training image pair entry from the name of a standar image in TestImages
"""
function generate_synthetic_noise_entry(imname :: String, params :: NamedTuple)
    im_true = Float64.(Gray.(TestImages.testimage(imname)))
    dynrange = maximum(im_true)
    im_noisy = im_true .+ params.noise_level.*randn(size(im_true)...)
    dim = size(im_true)
    return Entry(im_true,im_noisy,imname,dim,dynrange)
end


"""
    generate_dataset(imname, params)

Generate a dataset consisting on the number of entries defined using a test image in the TestImages.jl module
"""
function generate_dataset(imname :: String, params :: NamedTuple)
    dataset_name = imname
    entries = Entry[]
    for i = 1:params.num_entries
        entry = generate_synthetic_noise_entry(imname,params)
        push!(entries,entry)
    end
    return Dataset(dataset_name,entries)
end

end