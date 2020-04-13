##################
# Generating scalar ROF cost function
##################

__precompile__()

using PGFPlotsX

using AlgTools.Util
using AlgTools.LinkedLists
using ImageTools.Denoise
using ImageTools.Visualise

using BilevelToolbox.DatasetGen
using BilevelToolbox.Denoise


const dataset_params = (
    dataset_name = "lena_gray_256",
    noise_level = 0.1,
    num_entries = 1
)

dataset = generate_dataset(dataset_params.dataset_name,dataset_params)

const λ_range = 1:0.5:50

costs = Float64[]

for l = λ_range
    print(" $l\n")
    # Parameter definition
    default_params = (
        λ = l,
        τ = 0.01,
        verbose_iter = 1000,
        maxiter = 1000,
        save_results = false,
        save_iterations = false,
        verbose = false
    )
    params = default_params
    st, iterate = initialise_visualisation(false)
    # Denoise image
    x,y,st = denoise_rof_pdhg(dataset.entries[1].im_noisy; xinit=dataset.entries[1].im_noisy, iterate=iterate, params=params)
    cost = norm₂²(x-dataset.entries[1].im_true)/2
    push!(costs,cost)
end

#######################
# Plotting
#######################

p = @pgf Axis(
    {
        xmajorgrids,
        ymajorgrids
    },
    Plot(
        {
            no_marks,
        },
        Coordinates(λ_range,costs)
    )
)

print_tex(p)
pgfsave("rof_scalar_cost.pdf", p; include_preamble=true, dpi=200)
pgfsave("rof_scalar_cost.tex", p; include_preamble=true, dpi=200)