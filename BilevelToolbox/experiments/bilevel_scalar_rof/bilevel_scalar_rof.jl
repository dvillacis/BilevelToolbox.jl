##################
# Finding optimal parameter for scalar ROF Denoising
##################

__precompile__()

using AlgTools.Util
using AlgTools.LinkedLists
using ImageTools.Denoise
using ImageTools.Visualise

using BilevelToolbox.DatasetGen
using BilevelToolbox.Denoise

Image = Array{Float64,2}

const dataset_params = (
    dataset_name = "lena_gray_256",
    noise_level = 0.1,
    num_entries = 1
)

dataset = generate_dataset(dataset_params.dataset_name,dataset_params)

const bilevel_params = (
    λ_init = 6.0, # Initial parameter value
    η_1 = 0.3,
    η_2 = 0.9,
    γ_1 = 0.5,
    γ_2 = 2.0,
    radius_init = 1.0
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

function upper_level_gradient(x :: Image, λ :: Float64)
    return convert(Float64,rand(-1:1))
end

function solve_tr_subproblem(λ :: Float64, radius :: Float64, grad :: Float64)
    step = -radius*sign(grad)
    return step
end


#######################
# Nonsmooth Trust Region Algorithm
#######################
λ = bilevel_params.λ_init
radius = bilevel_params.radius_init
b = dataset.entries[1].im_noisy
x̄ = dataset.entries[1].im_true
for it = 1:10
    x = lower_level_solver(b,λ)
    cost = upper_level_cost(x,x̄)
    grad = upper_level_gradient(x,λ)
    step = solve_tr_subproblem(λ,radius,grad)
    print("$it: \tcost= $cost, λ= $λ, radius= $radius, grad= $grad")
    
    # Quality indicators
    pred = -grad*step
    ρ = 0
    if pred > 0
        x_ = lower_level_solver(b,λ+step)
        cost_ = upper_level_gradient(x,λ+step)
        ρ = (cost-cost_)/pred
    end
    print(", ρ = $ρ\n")
    #Trust-Region radius change
    if ρ > bilevel_params.η_2
        global λ += step
        global radius *= bilevel_params.γ_2
    elseif ρ <= bilevel_params.η_1
        global radius *= bilevel_params.γ_1
    else
        global λ += step
        global radius *= bilevel_params.γ_1
    end


end