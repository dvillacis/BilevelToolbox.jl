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

const default_save_prefix="denoise_result_"

const default_params = (
    α = 1,
    τ₀ = 5,
    σ₀ = 0.99/5,
    ρ = 0,
    accel = true,
    noise_level = 0.5,
    verbose_iter = 10,
    maxiter = 1000,
    save_results = false,
    image_name = "lighthouse",
    save_iterations = false
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

    # Load image and add noise
    b = Float64.(Gray{Float64}.(TestImages.testimage(params.image_name)))
    b_noisy = b .+ params.noise_level.*randn(size(b)...)

    # Launch (background) visualiser
    st, iterate = initialise_visualisation(visualise)

    # Run algorithm
    x, y, st = denoise_fista(b_noisy; iterate=iterate, params=params)

    if params.save_results
        perffile = params.save_prefix * ".txt"
        println("Saving " * perffile)
        write_log(perffile, st.log, "# params = $(params)\n")
        fn = (t, ext) -> "$(params.save_prefix)_$(t).$(ext)"
        save(File(format"PNG", fn("true", "png")), grayimg(b))
        save(File(format"PNG", fn("data", "png")), grayimg(b_noisy))
        save(File(format"PNG", fn("reco", "png")), grayimg(x))
    end

    # Exit background visualiser
    finalise_visualisation(st)
end
