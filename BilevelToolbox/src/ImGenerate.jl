###################
# Image generation
###################

module ImGenerate

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

export  ImGen,
        generate_noisy_image

##################
# Data structures
##################

struct ImGen
    im_true :: Image
    im_noisy :: Image
    name :: String
    dim :: Tuple{Int64,Int64}
    dynrange :: Float64
end

function generate_noisy_image(imname :: String, params :: NamedTuple)
    im_true = Float64.(Gray.(TestImages.testimage(imname)))
    dynrange = maximum(im_true)
    im_noisy = im_true .+ params.noise_level.*randn(size(im_true)...)
    dim = size(im_true)
    return ImGen(im_true,im_noisy,imname,dim,dynrange)
end

end