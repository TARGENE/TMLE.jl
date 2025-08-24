abstract type Estimator end

#####################################################################
###               MLConditionalDistributionEstimator              ###
#####################################################################

@auto_hash_equals struct MLConditionalDistributionEstimator <: Estimator
    model::MLJBase.Supervised
    train_validation_indices
    prevalence_weights::Union{Nothing, Vector{Float64}}
end

MLConditionalDistributionEstimator(models; train_validation_indices=nothing, prevalence_weights=nothing) = 
    MLConditionalDistributionEstimator(models, train_validation_indices, prevalence_weights)

MLConditionalDistributionEstimator(models, train_validation_indices; prevalence_weights=nothing) = 
    MLConditionalDistributionEstimator(models, train_validation_indices, prevalence_weights)

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

"""
    fit_mlj_model(model, X, y; parents=names(X), cache=false, weights=nothing, verbosity=1)
Fits a MLJ model to the data X and y, using the specified model.
- If the model has no parents, it is assumed to be a marginal model and a constant classifier is used.
- If weights are provided in the case of a case-control study, they are used to fit the model.
"""
function fit_mlj_model(model, X, y; parents=names(X), cache=false, weights=nothing, verbosity=1,)
    model = actual_model(model, parents, y)

    if isnothing(weights)
        mach = machine(model, X, y; cache=cache)
    else
        if supervised_learner_supports_weights(model)
            mach = machine(model, X, y, weights; cache=cache)
        else
            throw(ArgumentError("The model $(model) does not support weights and cannot be used with prevalence."))
        end
    end
    MLJBase.fit!(mach, verbosity=verbosity)
    return mach
end

"""
    compute_prevalence_weights(prevalence, y)

Calculates weights for a case-control study to use in the fitting of nuisance functions.
- `prevalence`: The prevalence of the outcome in the population.
- `y`: The outcome variable across observations, which should be binary vector.`
"""
function compute_prevalence_weights(prevalence::Float64, y::AbstractVector)
    J = sum(y .== 0) ÷ sum(y .== 1)
    weights = Vector{Float64}(undef, length(y))
    for i in eachindex(y)
        weights[i] = y[i] == 1 ? prevalence : (1 - prevalence) / J
    end
    return weights
end

compute_prevalence_weights(::Nothing, y) = nothing

get_training_prevalence_weights(::Nothing, train_indices) = nothing

get_training_prevalence_weights(weights::AbstractVector, ::Nothing) = weights

get_training_prevalence_weights(weights::AbstractVector, train_indices::Tuple) = weights[train_indices[1]]

get_training_prevalence_weights(weights::AbstractVector, train_indices::AbstractVector) = weights[train_indices]

"""
    get_matched_controls(dataset, relevant_factors, J)

Returns the matched controls for each case in the dataset based on the intended number of controls per case (J).
Randomly discards unmatched controls.

Currently, this implementation is for independent case-control studies. Will be expanded for matched case-control studies in the future.
"""
function get_matched_controls(dataset, relevant_factors; rng=Random.GLOBAL_RNG)
    y = dataset[!, relevant_factors.outcome_mean.outcome]
    # Choose integer J (floor(nCo/nC))
    J = sum(y .== 0) ÷ sum(y .== 1)
    idx_case = findall(y .== 1)
    idx_ctl  = findall(y .== 0)
    nC  = length(idx_case)
    nCo = length(idx_ctl)
    @assert nC > 0 "No cases found"
    @assert nCo >= nC * J "Not enough controls: need $(nC*J), have $nCo"
    ctl_pool = copy(idx_ctl)
    Random.shuffle!(rng, ctl_pool)
    sel_ctl = ctl_pool[1:(Int(nC*J))]
    keep_idx = sort!(vcat(idx_case, sel_ctl))
    return dataset[keep_idx, :]
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
    # If a prevalence weights are provided, we use it to fit the model
    weights = get_training_prevalence_weights(estimator.prevalence_weights, estimator.train_validation_indices)
    
    mach = fit_mlj_model(estimator.model, X, y; 
        parents=estimand.parents, 
        cache=machine_cache,
        weights=weights,
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
    prevalence_weights::Union{Nothing, Vector{Float64}}
end

SampleSplitMLConditionalDistributionEstimator(model, train_validation_indices; prevalence_weights=nothing) =
    SampleSplitMLConditionalDistributionEstimator(model, train_validation_indices, prevalence_weights)

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
    
    weights = get_training_prevalence_weights(estimator.prevalence_weights, train_indices)
    machines[fold_id] = fit_mlj_model(estimator.model, Xtrain, ytrain; 
        parents=estimand.parents, 
        cache=machine_cache,
        weights=weights, 
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
        verbosity=verbosity-1
        )
    # Build estimate
    estimate = SampleSplitMLConditionalDistribution(estimand, estimator.train_validation_indices, machines)
    # Update cache
    update_cache!(cache, estimand, estimator, estimate)

    return estimate
end

ConditionalDistributionEstimator(model, train_validation_indices::Union{Nothing,Tuple}; prevalence_weights=nothing) =
    MLConditionalDistributionEstimator(model, train_validation_indices, prevalence_weights)

ConditionalDistributionEstimator(model, train_validation_indices::AbstractVector; prevalence_weights=nothing) =
    SampleSplitMLConditionalDistributionEstimator(model, train_validation_indices, prevalence_weights)

    
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
    compose(f, Ψ̂::JointEstimate, backend)

Provides an estimator of f(Ψ̂).

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
- Ψ̂: A JointEstimate
- backend: A differentiation backend, e.g., AutoZygote(), AutoEnzyme(), etc.
"""
function compose end
    
compose(f, Ψ̂::JointEstimate, backend::Nothing=nothing) = 
    throw(ArgumentError("""In order to compose elements of a JointEstimate, you need to load `DifferentiationInterface` and a backend of your choice, e.g., `Zygote`. 
    Then call `compose(f, Ψ̂, backend)`, for example: `using DifferentiationInterface, Zygote; compose(f, Ψ̂, AutoZygote())`"""))

function covariance_matrix(estimates...)
    X = hcat([r.IC for r in estimates]...)
    return cov(X, dims=1, corrected=true)
end