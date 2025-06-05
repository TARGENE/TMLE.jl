abstract type Estimator end

#####################################################################
###               MLConditionalDistributionEstimator              ###
#####################################################################

@auto_hash_equals struct MLConditionalDistributionEstimator <: Estimator
    model::MLJBase.Supervised
    train_validation_indices
end

MLConditionalDistributionEstimator(models; train_validation_indices=nothing) = 
    MLConditionalDistributionEstimator(models, train_validation_indices)

marginal_model(y::CategoricalVector) = ConstantClassifier()

marginal_model(y::AbstractVector) = throw(ArgumentError(string("Marginal model not implemented for non categorical targets, type of the target: ", eltype(y))))

"""
    actual_model(model, parents, y)

If there are no parents, we are trying to fit a marginal distribution, 
then we return a constant classifier. Otherwise we return the user-specified model.

This situation can arise in at least two cases for the propensity score:
- Initialisation of some CTMLE strategies
- There is actually no confounders

Since there is no current understanding of when this could arise for a continuous outcome y, we throw an error if y is not categorical.
"""
actual_model(model, parents, y) =
    isempty(parents) ? marginal_model(y) : model

function fit_mlj_model(model, X, y; parents=names(X), cache=false, verbosity=1)
    model = actual_model(model, parents, y)
    mach = machine(model, X, y, cache=cache)
    MLJBase.fit!(mach, verbosity=verbosity)
    return mach
end

function (estimator::MLConditionalDistributionEstimator)(estimand, dataset; 
    cache=Dict(), 
    verbosity=1, 
    machine_cache=false,
    acceleration=CPU1()
    )
    # Lookup in cache
    estimate = estimate_from_cache(cache, estimand, estimator; verbosity=verbosity)
    estimate !== nothing && return estimate

    verbosity > 0 && @info(string("Estimating: ", string_repr(estimand)))
    # Otherwise estimate
    relevant_dataset = nomissing(dataset, variables(estimand))
    relevant_dataset = training_rows(relevant_dataset, estimator.train_validation_indices)
    # Fit Conditional DIstribution using MLJ
    X = TMLE.selectcols(relevant_dataset, estimand.parents)
    y = relevant_dataset[!, estimand.outcome]
    mach = fit_mlj_model(estimator.model, X, y; 
        parents=estimand.parents, 
        cache=machine_cache, 
        verbosity=verbosity-1
    )
    # Build estimate
    estimate = MLConditionalDistribution(estimand, mach)
    # Update cache
    update_cache!(cache, estimand, estimator, estimate)

    return estimate
end

training_rows(dataset, train_validation_indices) = selectrows(dataset, train_validation_indices[1])

training_rows(dataset, train_validation_indices::Nothing) = dataset

#####################################################################
###       SampleSplitMLConditionalDistributionEstimator           ###
#####################################################################

"""
Estimates a conditional distribution (or regression) for each training set defined by `train_validation_indices`.
"""
@auto_hash_equals struct SampleSplitMLConditionalDistributionEstimator <: Estimator
    model::MLJBase.Supervised
    train_validation_indices
end

function update_sample_split_machines_with_fold!(machines::Vector{Machine}, 
    estimator, 
    estimand, 
    dataset, 
    fold_id;
    machine_cache=false, 
    verbosity=1
    )
    train_indices, _ = estimator.train_validation_indices[fold_id]
    train_dataset = selectrows(dataset, train_indices)
    Xtrain = selectcols(train_dataset, estimand.parents)
    ytrain = train_dataset[!, estimand.outcome]
    machines[fold_id] = fit_mlj_model(estimator.model, Xtrain, ytrain; 
        parents=estimand.parents, 
        cache=machine_cache, 
        verbosity=verbosity
    )
end

function fit_sample_split_machines!(machines::Vector{Machine}, acceleration::CPU1, estimator, estimand, dataset;
    machine_cache=false, 
    verbosity=1
    )
    nfolds = length(machines)
    for fold_id in 1:nfolds
        update_sample_split_machines_with_fold!(machines,
            estimator, 
            estimand, 
            dataset, 
            fold_id;
            machine_cache=machine_cache, 
            verbosity=verbosity
        )
    end
end

function fit_sample_split_machines!(machines::Vector{Machine}, acceleration::CPUThreads, estimator, estimand, dataset;
    machine_cache=false, 
    verbosity=1
    )
    nfolds = length(machines)
    @threads for fold_id in 1:nfolds
        update_sample_split_machines_with_fold!(machines,
            estimator, 
            estimand, 
            dataset, 
            fold_id;
            machine_cache=machine_cache, 
            verbosity=verbosity
        )
    end
end

function (estimator::SampleSplitMLConditionalDistributionEstimator)(estimand, dataset; 
    cache=Dict(), 
    verbosity=1, 
    machine_cache=false,
    acceleration=CPU1()
    )
    # Lookup in cache
    estimate = estimate_from_cache(cache, estimand, estimator; verbosity=verbosity)
    estimate !== nothing && return estimate

    # Otherwise estimate
    verbosity > 0 && @info(string("Estimating: ", string_repr(estimand)))
    
    relevant_dataset = selectcols(dataset, variables(estimand))
    nfolds = size(estimator.train_validation_indices, 1)
    machines = Vector{Machine}(undef, nfolds)
    # Fit Conditional Distribution on each training split using MLJ
    fit_sample_split_machines!(machines, acceleration, estimator, estimand, relevant_dataset;
        machine_cache=machine_cache, 
        verbosity=verbosity-1,
        )
    # Build estimate
    estimate = SampleSplitMLConditionalDistribution(estimand, estimator.train_validation_indices, machines)
    # Update cache
    update_cache!(cache, estimand, estimator, estimate)

    return estimate
end

ConditionalDistributionEstimator(model, train_validation_indices::Union{Nothing,Tuple}) =
    MLConditionalDistributionEstimator(model, train_validation_indices)

ConditionalDistributionEstimator(model, train_validation_indices::AbstractVector) =
    SampleSplitMLConditionalDistributionEstimator(model, train_validation_indices)

    
#####################################################################
###            JointConditionalDistributionEstimator              ###
#####################################################################

@auto_hash_equals struct JointConditionalDistributionEstimator <: Estimator 
    cd_estimators::Dict{Symbol, Any}
end

function fit_conditional_distributions(acceleration::CPU1, cd_estimators, conditional_distributions, dataset; cache=Dict(), verbosity=1, machine_cache=false)
    return map(conditional_distributions) do conditional_distribution
        cd_estimator = cd_estimators[conditional_distribution.outcome]
        try_fit_ml_estimator(cd_estimator, conditional_distribution, dataset;
            error_fn=propensity_score_fit_error_msg,
            cache=cache,
            verbosity=verbosity,
            machine_cache=machine_cache,
            acceleration=acceleration
        )
    end
end

function fit_conditional_distributions(acceleration::CPUThreads, cd_estimators, conditional_distributions, dataset; 
    cache=Dict(),
    verbosity=1, 
    machine_cache=false
    )
    n_components = length(conditional_distributions)
    estimates = Vector{ConditionalDistributionEstimate}(undef, n_components)
    @threads for cd_index in 1:n_components
        conditional_distribution = conditional_distributions[cd_index]
        cd_estimator = cd_estimators[conditional_distribution.outcome]
        estimates[cd_index] = try_fit_ml_estimator(cd_estimator, conditional_distribution, dataset;
            error_fn=propensity_score_fit_error_msg,
            cache=cache,
            verbosity=verbosity,
            machine_cache=machine_cache,
            acceleration=acceleration
        )
    end
    return Tuple(estimates)
end

function (estimator::JointConditionalDistributionEstimator)(conditional_distributions, dataset; 
    cache=Dict(), 
    verbosity=1, 
    machine_cache=false,
    acceleration=CPU1()
    )
    estimates = fit_conditional_distributions(acceleration, estimator.cd_estimators, conditional_distributions, dataset; 
        cache=cache, 
        verbosity=verbosity, 
        machine_cache=machine_cache
    )
    return JointConditionalDistributionEstimate(conditional_distributions, estimates)
end

#####################################################################
###                    JointEstimand Estimator                    ###
#####################################################################

"""
    (estimator::Estimator)(Ψ::JointEstimand, dataset; cache=Dict(), verbosity=1)

Estimates all components of Ψ and then Ψ itself.
"""
function (estimator::Estimator)(Ψ::JointEstimand, dataset; cache=Dict(), verbosity=1, acceleration=CPU1())
    estimates = map(Ψ.args) do estimand 
        estimate, _ = estimator(estimand, dataset; cache=cache, verbosity=verbosity, acceleration=acceleration)
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