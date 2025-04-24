abstract type Estimator end

#####################################################################
###            MLFoldsConditionalDistributionEstimator            ###
#####################################################################

"""
    MLFoldsConditionalDistributionEstimator

Estimates  conditional distribution (or regression) for dataset's subset identified by `fold`.
"""
@auto_hash_equals struct MLFoldsConditionalDistributionEstimator{T} <: Estimator
    model::MLJBase.Supervised
    train_validation_indices::Tuple
end


#####################################################################
###               MLConditionalDistributionEstimator              ###
#####################################################################

@auto_hash_equals struct MLConditionalDistributionEstimator <: Estimator
    model::MLJBase.Supervised
end

function (estimator::MLConditionalDistributionEstimator)(estimand, dataset; cache=Dict(), verbosity=1, machine_cache=false)
    # Lookup in cache
    estimate = estimate_from_cache(cache, estimand, estimator; verbosity=verbosity)
    estimate !== nothing && return estimate

    verbosity > 0 && @info(string("Estimating: ", string_repr(estimand)))
    # Otherwise estimate
    relevant_dataset = nomissing(dataset, variables(estimand))
    # Fit Conditional DIstribution using MLJ
    X = selectcols(relevant_dataset, estimand.parents)
    y = Tables.getcolumn(relevant_dataset, estimand.outcome)
    mach = machine(estimator.model, X, y, cache=machine_cache)
    fit!(mach, verbosity=verbosity-1)
    # Build estimate
    estimate = MLConditionalDistribution(estimand, mach)
    # Update cache
    update_cache!(cache, estimand, estimator, estimate)

    return estimate
end

key(estimator::MLConditionalDistributionEstimator) =
    (MLConditionalDistributionEstimator, estimator.model)

#####################################################################
###       SampleSplitMLConditionalDistributionEstimator           ###
#####################################################################

"""
Estimates a conditional distribution (or regression) for each training set defined by `train_validation_indices`.
"""
@auto_hash_equals struct SampleSplitMLConditionalDistributionEstimator <: Estimator
    model::MLJBase.Supervised
    train_validation_indices::Tuple
end

function (estimator::SampleSplitMLConditionalDistributionEstimator)(estimand, dataset; cache=Dict(), verbosity=1, machine_cache=false)
    # Lookup in cache
    estimate = estimate_from_cache(cache, estimand, estimator; verbosity=verbosity)
    estimate !== nothing && return estimate

    # Otherwise estimate
    verbosity > 0 && @info(string("Estimating: ", string_repr(estimand)))
    
    relevant_dataset = selectcols(dataset, variables(estimand))
    nfolds = size(estimator.train_validation_indices, 1)
    machines = Vector{Machine}(undef, nfolds)
    # Fit Conditional Distribution on each training split using MLJ
    for (index, (train_indices, _)) in enumerate(estimator.train_validation_indices)
        train_dataset = selectrows(relevant_dataset, train_indices)
        Xtrain = selectcols(train_dataset, estimand.parents)
        ytrain = Tables.getcolumn(train_dataset, estimand.outcome)
        mach = machine(estimator.model, Xtrain, ytrain, cache=machine_cache)
        fit!(mach, verbosity=verbosity-1)
        machines[index] = mach
    end
    # Build estimate
    estimate = SampleSplitMLConditionalDistribution(estimand, estimator.train_validation_indices, machines)
    # Update cache
    update_cache!(cache, estimand, estimator, estimate)

    return estimate
end

key(estimator::SampleSplitMLConditionalDistributionEstimator) =
    (MLConditionalDistributionEstimator, estimator.model, estimator.train_validation_indices)

ConditionalDistributionEstimator(model, train_validation_indices::Nothing) =
    MLConditionalDistributionEstimator(model)

ConditionalDistributionEstimator(model, train_validation_indices) =
    SampleSplitMLConditionalDistributionEstimator(model, train_validation_indices)

    
#####################################################################
###            JointConditionalDistributionEstimator              ###
#####################################################################

@auto_hash_equals struct JointConditionalDistributionEstimator <: Estimator 
    cd_estimators::Dict{Symbol, Any}
end

function fit_conditional_distributions(cd_estimators, conditional_distributions, dataset; cache=Dict(), verbosity=1, machine_cache=false)
    return map(conditional_distributions) do conditional_distribution
        cd_estimator = cd_estimators[conditional_distribution.outcome]
        try_fit_ml_estimator(cd_estimator, conditional_distribution, dataset;
            error_fn=propensity_score_fit_error_msg,
            cache=cache,
            verbosity=verbosity,
            machine_cache=machine_cache,
        )
    end
end

function (estimator::JointConditionalDistributionEstimator)(conditional_distributions, dataset; 
    cache=Dict(), 
    verbosity=1, 
    machine_cache=false
    )
    estimates = fit_conditional_distributions(estimator.cd_estimators, conditional_distributions, dataset; 
        cache=cache, 
        verbosity=verbosity, 
        machine_cache=machine_cache
    )
    return JointConditionalDistributionEstimate(conditional_distributions, estimates)
end

#####################################################################
###                   JointEstimand Estimator                  ###
#####################################################################

"""
    (estimator::Estimator)(Ψ::JointEstimand, dataset; cache=Dict(), verbosity=1)

Estimates all components of Ψ and then Ψ itself.
"""
function (estimator::Estimator)(Ψ::JointEstimand, dataset; cache=Dict(), verbosity=1)
    estimates = map(Ψ.args) do estimand 
        estimate, _ = estimator(estimand, dataset; cache=cache, verbosity=verbosity)
        estimate
    end
    Σ = covariance_matrix(estimates...)
    n = size(first(estimates).IC, 1)
    return JointEstimate(Ψ, estimates, Σ, n), cache
end

"""
    joint_estimand(args...)
    
This function is temporary and only necessary because of a bug in 
the AD package. Simply call `vcat` in the future.
"""
joint_estimand(args...) = vcat(args...)

"""
    compose(f, estimation_results::Vararg{EICEstimate, N}) where N

Provides an estimator of f(estimation_results...).

# Mathematical details

The following is a summary from `Asymptotic Statistics`, A. W. van der Vaart.

Consider k TMLEs computed from a dataset of size n and embodied by Tₙ = (T₁,ₙ, ..., Tₖ,ₙ). 
Since each of them is asymptotically normal, the multivariate CLT provides the joint 
distribution:

    √n(Tₙ - Ψ₀) ↝ N(0, Σ), 
    
where Σ is the covariance matrix of the TMLEs influence curves.

Let f:ℜᵏ→ℜᵐ, be a differentiable map at Ψ₀. Then, the delta method provides the
limiting distribution of √n(f(Tₙ) - f(Ψ₀)). Because Tₙ is normal, the result is:

    √n(f(Tₙ) - f(Ψ₀)) ↝ N(0, ∇f(Ψ₀) ̇Σ ̇(∇f(Ψ₀))ᵀ),

where ∇f(Ψ₀):ℜᵏ→ℜᵐ is a linear map such that by abusing notations and identifying the 
function with the multiplication matrix: ∇f(Ψ₀):h ↦ ∇f(Ψ₀) ̇h. And the matrix ∇f(Ψ₀) is 
the jacobian of f at Ψ₀.

Hence, the only thing we need to do is:
- Compute the covariance matrix Σ
- Compute the jacobian ∇f, which can be done using Julia's automatic differentiation facilities.
- The final estimator is normal with mean f₀=f(Ψ₀) and variance σ₀=∇f(Ψ₀) ̇Σ ̇(∇f(Ψ₀))ᵀ

# Arguments

- f: An array-input differentiable map.
- estimation_results: 1 or more `EICEstimate` structs.

# Examples

Assuming `res₁` and `res₂` are TMLEs:

```julia
f(x, y) = [x^2 - y, y - 3x]
compose(f, res₁, res₂)
```
"""
function compose(f, Ψ̂::JointEstimate; backend=AutoZygote())
    point_estimate = estimate(Ψ̂)
    Σ = Ψ̂.cov
    f₀, J = value_and_jacobian(f, backend, point_estimate)
    σ₀ = J * Σ * J'
    estimand = ComposedEstimand(f, Ψ̂.estimand)
    return ComposedEstimate(estimand, f₀, σ₀, Ψ̂.n)
end

function covariance_matrix(estimates...)
    X = hcat([r.IC for r in estimates]...)
    return cov(X, dims=1, corrected=true)
end