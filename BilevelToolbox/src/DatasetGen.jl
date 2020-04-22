###################
# Image generation
###################

module DatasetGen

using ColorTypes: Gray
using FileIO
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

export  AbstractDataset,
        AbstractEntry,
        Dataset,
        DirDataset,
        Entry,
        DirEntry,
        generate_dataset,
        generate_dir_dataset,
        generate_synthetic_noise_entry,
        get_training_pair

##################
# Data structures
##################

abstract type AbstractDataset end
abstract type AbstractEntry end

struct Entry <: AbstractEntry
    im_true :: Image
    im_noisy :: Image
    name :: String
    dim :: Tuple{Int64,Int64}
    dynrange :: Float64
end

struct DirEntry <: AbstractEntry
    im_true :: String
    im_noisy :: String
end

struct Dataset <: AbstractDataset
    name :: String
    entries :: Array{Entry,1}
end

struct DirDataset <: AbstractDataset
    name :: String
    path :: String
    entries :: Array{DirEntry,1}
end

function Base.show(io::IO, e::Entry)
    print(io,"
        name: $(e.name)
        dim: $(e.dim)
    ")
end

function Base.show(io::IO, e::DirEntry)
    print(io,"
        im_true: $(e.im_true)
        im_noisy: $(e.im_noisy)
    ")
end

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


function generate_dir_dataset(dataset_path :: String, params :: NamedTuple)
    entries = DirEntry[]
    if isdir(dataset_path)
        for (root,dirs,files) in walkdir(dataset_path)
            for file in files
                if occursin(params.true_regex,file)
                    im_true = file
                    num = split(file,"_")[1]
                    ext = split(file,".")[end]
                    im_noisy = num*"_"*params.noisy_regex*"."*ext
                    entry = DirEntry(im_true,im_noisy)
                    push!(entries,entry)
                end
            end
        end
        return DirDataset(params.dataset_name,dataset_path,entries)
    else
        println("dataset_path does not exists. Exiting...")
    end
end

function get_training_pair(idx :: Integer, dataset::Dataset)
    entry = dataset.entries[idx]
    return (entry.im_true,entry.im_noisy)
end

function get_training_pair(idx :: Integer, dataset::DirDataset)
    entry = dataset.entries[idx]
    im_true = Float64.(load(dataset.path*"/"*entry.im_true))
    im_noisy = Float64.(load(dataset.path*"/"*entry.im_noisy))
    return (im_true,im_noisy)
end

end