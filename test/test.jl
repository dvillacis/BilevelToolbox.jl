##################
# Denoise testing
##################

__precompile__()

using Printf
using FileIO
using ColorTypes: Gray
# ColorVectorSpace is only needed to ensure that conversions
# between different ColorTypes are defined.
import ColorVectorSpace
import TestImages

using AlgTools.Util
using AlgTools.LinkedLists
using ImageTools.Denoise
using ImageTools.Visualise

using BilevelToolbox.DatasetGen
using BilevelToolbox.Denoise

const default_save_prefix="denoise_result_"

const default_params = (
    λ = 5,
    τ = 0.01,
    noise_level = 0.1,
    verbose_iter = 100,
    maxiter = 1000,
    save_results = true,
    image_name = "lena_gray_512",
    save_iterations = false
)

const dataset_params = (
    dataset_name = "lena",
    num_entries = 1
)

#######################
# Main testing routine
#######################

function test_denoise(;
                      visualise=true,
                      save_prefix=default_save_prefix,
                      kwargs...)

    
    # Parameters for this experiment
    params = default_params ⬿ kwargs
    params = params ⬿ (save_prefix = save_prefix * params.image_name,)
    params = default_params ⬿ dataset_params

    save_prefix = "denoise_result_" * params.image_name

    # Generate dataset entry 
    dataset = generate_dataset(params.image_name,params)

    # Launch (background) visualiser
    st, iterate = initialise_visualisation(visualise)

    # Run algorithm
    x, y, st = denoise_rof_pdhg(dataset.entries[1].im_noisy; xinit=dataset.entries[1].im_noisy, iterate=iterate, params=params)

    if params.save_results
        #perffile = params.save_prefix * ".txt"
        perffile = save_prefix * ".txt"
        println("Saving " * perffile)
        write_log(perffile, st.log, "# params = $(params)\n")
        #fn = (t, ext) -> "$(params.save_prefix)_$(t).$(ext)"
        fn = (t, ext) -> "$(save_prefix)_$(t).$(ext)"
        save(File(format"PNG", fn("true", "png")), grayimg(dataset.entries[1].im_true))
        save(File(format"PNG", fn("data", "png")), grayimg(dataset.entries[1].im_noisy))
        save(File(format"PNG", fn("reco", "png")), grayimg(x))
    end

    # Exit background visualiser
    finalise_visualisation(st)
end

test_denoise()