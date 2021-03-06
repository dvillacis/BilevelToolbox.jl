####################################################################
# Nonsmooth Trust Region Algorithm
####################################################################

__precompile__()

module AlgorithmTrustRegionNS

using LinearAlgebra
using Printf
using ColorTypes: Gray

using AlgTools.Util
import AlgTools.Iterate
using ImageTools.Gradient
using ImageTools.Translate
using ImageTools.Visualise

using BilevelToolbox.DatasetGen
using BilevelToolbox.BilevelIterate

###############################################
# Types (several imported from ImageTools.Translate)
###############################################

Image = Translate.Image


############
# DOGBOX
############

"""
Find intersection of trust region bounds and initial bounds
"""
function find_intersection(x,tr_bounds, lb,ub)
    lb_centered = lb - x
    ub_centered = ub - x

    lb_total = max.(lb_centered, -tr_bounds)
    ub_total = min.(ub_centered, tr_bounds)

    orig_l = lb_total .== lb_centered
    orig_u = ub_total .== ub_centered

    tr_l = lb_total .== -tr_bounds
    tr_u = ub_total .== tr_bounds

    return lb_total, ub_total, orig_l, orig_u, tr_l, tr_u
end

"""
Check if a point lies within bounds
"""
function in_bounds(x, lb, ub)
    return all((x .>= lb) .& (x .<= ub))
end

"""
Compute a min step required to reach a bound.
This function computes a positive scalar t, such that x + s*t is on the bound
"""
function step_size_to_bound(x,s,lb,ub)
    non_zero = findall(s .!= 0)
    s_non_zero = s[non_zero]
    steps = zeros(size(x)...)
    steps[non_zero] = max.((lb - x)[non_zero]/s_non_zero, (ub - x)[non_zero]/s_non_zero)
    min_step = min(steps)
    return min_step, steps .== min_step .* sign.(s)
end

"""
Find the dogleg step in a rectangular region
"""
function dogleg_step(x, newton_step, g, a, b, tr_bounds, lb, ub)

    lb_total, ub_total, orig_l, orig_u, tr_l, tr_u = find_intersection(x,tr_bounds,lb,ub)
    
    bound_hits = zeros(size(x)...)
    if in_bounds(newton_step, lb_total, ub_total)
        return newton_step, bound_hits, false # If newton step satisfies bounds, then take it!
    end

    to_bounds, _ = step_size_to_bound(zeros(size(x)),-g, lb_total, ub_total)


end


############
# Auxiliary functions
############

function solve_tr_subproblem(λ :: Float64, radius :: Float64, grad :: Float64, H :: Float64)
    # classic Cauchy point calculation
    s = 1
    if grad*H*grad > 0
        s = min(norm(grad)^3/(radius*grad*H*grad),1)
    end
    return -s*radius*sign(grad)
end

function bfgs_update(B,λ,λ_,grad,grad_)
    s = λ_ - λ
    Bs = B*s
    y = grad_ - grad
    if norm(y) > 0
        B = B + (y^2)/(y*s) - Bs*Bs/(Bs*s)
    end
    return B
end


############
# Algorithm
############

function solve(dataset :: AbstractDataset, lower_level_solver :: Function, upper_level_cost :: Function, upper_level_gradient :: Function; iterate=AlgTools.simple_iterate, params::NamedTuple)

    ################################                                        
    # Extract and set up parameters
    ################################
    η_1, η_2 = params.η_1, params.η_2   # Threshold values for quality indicator
    γ_1, γ_2 = params.γ_1, params.γ_2   # Radius increase / decrese parameters
    λ = params.λ_init                   # Initial parameter value
    radius = params.radius_init         # Initial radius value
    B = params.B_init                   # Inital second order approximation
    
    x̄,b = get_training_pair(1,dataset)  # TODO: the functions should call it instead


    ######################
    # Initialise iterates
    ######################
    x = copy(b)

    ####################
    # Run the algorithm
    ####################

    v = iterate(params) do verbose :: Function
        x = lower_level_solver(b,λ)
        cost = upper_level_cost(x,x̄)
        grad = upper_level_gradient(b,x̄,x,λ)
        step = solve_tr_subproblem(λ,radius,grad,B)
        pred = -grad*step - 0.5*step*B*step
        ρ = 0
        if pred > 0
            x_ = lower_level_solver(b,λ+step)
            cost_ = upper_level_cost(x_,x̄)
            ρ = (cost-cost_)/pred
        end
        if ρ > η_2
            x_ = lower_level_solver(b,λ+step)
            grad_ = upper_level_gradient(b,x̄,x_,λ+step)
            B = bfgs_update(B,λ,λ+step,grad,grad_)
            λ += step
            radius *= γ_2
        elseif ρ <= η_1
            radius *= γ_1
        else
            λ += step
            radius *= γ_1
        end
        ################################
        # Give function value if needed
        ################################
        v = verbose() do            
            cost, λ, radius, grad, B, ρ, x
        end
        v
    end
    return λ, x, v

end 

end # Module