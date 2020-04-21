##################
# Finding optimal parameter for scalar ROF Denoising
##################

__precompile__()

using LinearAlgebra

using AlgTools.Util
using AlgTools.LinkedLists
using ImageTools.Denoise
using ImageTools.Visualise

using BilevelToolbox
using BilevelToolbox.DatasetGen
using BilevelToolbox.Denoise
using BilevelToolbox.AlgorithmTrustRegionNS

Image = Array{Float64,2}

const dataset_params = (
    dataset_name = "lena_gray_256",
    noise_level = 0.1,
    num_entries = 1
)

dataset = generate_dataset(dataset_params.dataset_name,dataset_params)

const bilevel_params = (
    λ_init = 20.0, # Initial parameter value
    η_1 = 0.3,
    η_2 = 0.9,
    γ_1 = 0.5,
    γ_2 = 2.0,
    radius_init = 1.0, # Initial trust-region radius
    B_init = 0.1,   # Initial hessian
    verbose_iter = 1,
    maxiter = 10,
    tol = 1e-3,
    save_iterations = false,
    save_results = true
)

function lower_level_solver(b :: Image, λ :: Float64)
    denoise_params = (
        λ = λ,
        τ = 0.01,
        verbose_iter = 1001,
        maxiter = 1000,
        save_results = false,
        save_iterations = false,
        verbose = false
    )
    st, iterate = initialise_visualisation(false)
    # Denoise image
    x,y,st = denoise_rof_pdhg(b; xinit=b, iterate=iterate, params=denoise_params)
    return x
end

function upper_level_cost(x :: Image, x̄ :: Image)
    return norm₂²(x-x̄)/2
end

function upper_level_gradient(b :: Image, x̄ :: Image, x :: Image, λ :: Float64)
    # Forward difference gradient approximation
    x_ = lower_level_solver(b,λ+1e-7)
    cost =  upper_level_cost(x,x̄)
    cost_ =  upper_level_cost(x_,x̄)
    return (cost_-cost)/1e-7
end

# Define experiment
experiment = Experiment(AlgorithmTrustRegionNS,dataset,lower_level_solver,upper_level_cost,upper_level_gradient,bilevel_params)

# Run experiment
run_bilevel_algorithm(experiment)