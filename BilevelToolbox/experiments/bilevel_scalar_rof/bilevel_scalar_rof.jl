##################
# Finding optimal parameter for scalar ROF Denoising
##################

__precompile__()

using LinearAlgebra

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
    λ_init = 20.0, # Initial parameter value
    η_1 = 0.3,
    η_2 = 0.9,
    γ_1 = 0.5,
    γ_2 = 2.0,
    radius_init = 1.0, # Initial trust-region radius
    H_init = 0.1,   # Initial hessian
    max_iterations = 100,
    tol = 1e-3
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

function solve_tr_subproblem(λ :: Float64, radius :: Float64, grad :: Float64, H :: Float64)
    # classic Cauchy point calculation
    s = 1
    if grad*H*grad > 0
        s = min(norm(grad)^3/(radius*grad*H*grad),1)
    end
    return -s*radius*sign(grad)
end

function bfgs_update(H,λ,λ_,grad,grad_)
    s = λ_ - λ
    Hs = H*s
    y = grad_ - grad
    if norm(y) > 0
        H = H + (y^2)/(y*s) - Hs*Hs/(Hs*s)
    end
    return H
end


#######################
# Nonsmooth Trust Region Algorithm
#######################

# Algorithm parameters
λ = bilevel_params.λ_init
radius = bilevel_params.radius_init
b = dataset.entries[1].im_noisy
x̄ = dataset.entries[1].im_true
H = bilevel_params.H_init

#######################
# Algorithm Run
#######################

for it = 1:bilevel_params.max_iterations
    if radius < bilevel_params.tol
        break
    end
    x = lower_level_solver(b,λ)
    cost = upper_level_cost(x,x̄)
    grad = upper_level_gradient(b,x̄,x,λ)
    if norm(grad) < bilevel_params.tol
        break
    end
    step = solve_tr_subproblem(λ,radius,grad,H)
    print("$it: \tcost= $cost, λ= $λ, radius= $radius, grad= $grad")
    
    # Quality indicators
    pred = -grad*step - 0.5*step*H*step
    ρ = 0
    if pred > 0
        x_ = lower_level_solver(b,λ+step)
        cost_ = upper_level_cost(x_,x̄)
        ρ = (cost-cost_)/pred
    end
    print(", H = $H, ρ = $ρ\n")

    #Trust-Region radius change
    if ρ > bilevel_params.η_2
        x_ = lower_level_solver(b,λ+step)
        grad_ = upper_level_gradient(b,x̄,x_,λ+step)
        global H = bfgs_update(H,λ,λ+step,grad,grad_)
        global λ += step
        global radius *= bilevel_params.γ_2
    elseif ρ <= bilevel_params.η_1
        global radius *= bilevel_params.γ_1
    else
        global λ += step
        global radius *= bilevel_params.γ_1
    end
end