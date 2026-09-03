#####################################################################
###             FoldsCMRelevantFactorsEstimator                   ###
#####################################################################

@auto_hash_equals struct FoldsCMRelevantFactorsEstimator <: Estimator
    models::Dict
    train_validation_indices
end

FoldsCMRelevantFactorsEstimator(models; train_validation_indices=nothing) = 
    FoldsCMRelevantFactorsEstimator(models, train_validation_indices)

function (estimator::FoldsCMRelevantFactorsEstimator)(acceleration::CPU1, estimand, dataset;
    cache=Dict(), 
    verbosity=1, 
    machine_cache=false
    )
    estimates = []
    for train_validation_indices in estimator.train_validation_indices
        η̂ = CMRelevantFactorsEstimator(
            train_validation_indices=train_validation_indices, 
            models=estimator.models
        )
        η̂ₙ = η̂(estimand, dataset; cache=cache, verbosity=verbosity, machine_cache=machine_cache)
        push!(estimates, η̂ₙ)
    end
    return estimates
end

function (estimator::FoldsCMRelevantFactorsEstimator)(acceleration::CPUThreads, estimand, dataset;
    cache=Dict(), 
    verbosity=1, 
    machine_cache=false
    )
    nfolds = length(estimator.train_validation_indices)
    estimates = Vector{Any}(undef, nfolds)
    @threads for fold_index in 1:nfolds
        train_validation_indices = estimator.train_validation_indices[fold_index]
        η̂ = CMRelevantFactorsEstimator(
            train_validation_indices=train_validation_indices, 
            models=estimator.models
        )
        η̂ₙ = η̂(estimand, dataset; cache=cache, verbosity=verbosity, machine_cache=machine_cache)
        estimates[fold_index] = η̂ₙ
    end
    return estimates
end

function (estimator::FoldsCMRelevantFactorsEstimator)(estimand, dataset; 
    cache=Dict(), 
    verbosity=1, 
    machine_cache=false,
    acceleration=CPU1()
    )
    # Lookup in cache
    estimate = estimate_from_cache(cache, estimand, estimator; verbosity=verbosity)
    estimate !== nothing && return estimate

    # Otherwise estimate
    verbosity > 0 && @info(string("Required ", string_repr(estimand)))

    estimates = estimator(acceleration, estimand, dataset;
        cache=cache, 
        verbosity=verbosity, 
        machine_cache=machine_cache
    )

    # Build estimate
    estimate = FoldsMLCMRelevantFactors(estimand, estimates)
    # Update cache
    update_cache!(cache, estimand, estimator, estimate)

    return estimate
end

#####################################################################
###                  CMRelevantFactorsEstimator                   ###
#####################################################################

@auto_hash_equals struct CMRelevantFactorsEstimator <: Estimator
    train_validation_indices
    models::Dict
    prevalence_weights::Union{Nothing, Vector{Float64}}
end

CMRelevantFactorsEstimator(;models, train_validation_indices=nothing, prevalence_weights=nothing) = CMRelevantFactorsEstimator(train_validation_indices, models, prevalence_weights)

"""
If there is no collaborative strategy, we are in CV mode and `train_validation_indices` are used to build the initial estimator.
"""
CMRelevantFactorsEstimator(collaborative_strategy::Nothing; models, train_validation_indices=nothing, prevalence_weights=nothing) = CMRelevantFactorsEstimator(train_validation_indices, models, prevalence_weights)

"""
If there is a collaborative strategy, `train_validation_indices` are ignored to build the initial estimator.
"""
CMRelevantFactorsEstimator(collaborative_strategy; models, train_validation_indices=nothing, prevalence_weights=nothing) = CMRelevantFactorsEstimator(nothing, models, prevalence_weights)

"""
    acquire_model(models, key, dataset, default_model_key=nothing)

Look up a model from `models` for the variable `key`. If `key` is found directly, return it.
Otherwise fall back to `default_model_key`. When `default_model_key` is `nothing`, the default
is inferred from the data: `:Q_binary_default` or `:Q_continuous_default` depending on whether
`key` is binary in `dataset`.

If the default key is also missing from `models`, a `KeyError` is raised.
"""
function acquire_model(models, key, dataset, default_model_key::Union{Nothing, Symbol}=nothing)
    haskey(models, key) && return models[key]
    if default_model_key === nothing
        default_model_key = is_binary(dataset, key) ? :Q_binary_default : :Q_continuous_default
    end
    haskey(models, default_model_key) && return models[default_model_key]
    throw(KeyError(default_model_key))
end

function build_propensity_score_estimator(propensity_score, models, dataset;
    train_validation_indices=nothing,
    prevalence_weights=nothing
    )
    cd_estimators = Dict()
    for conditional_distribution in propensity_score
        outcome = conditional_distribution.outcome
        model = acquire_model(models, outcome, dataset, :G_default)
        cd_estimators[outcome] = ConditionalDistributionEstimator(model, train_validation_indices, prevalence_weights=prevalence_weights)
    end
    return JointConditionalDistributionEstimator(cd_estimators)
end

function estimate_propensity_score(propensity_score, models, dataset;
    train_validation_indices=nothing,
    cache=Dict(),
    verbosity=1,
    machine_cache=false,
    acceleration=CPU1(),
    prevalence_weights=nothing
    )
    propensity_score_estimator = build_propensity_score_estimator(
        propensity_score, 
        models,  
        dataset;
        train_validation_indices=train_validation_indices,
        prevalence_weights=prevalence_weights
    )
    return propensity_score_estimator(
        propensity_score, 
        dataset;
        cache=cache,
        verbosity=verbosity,
        machine_cache=machine_cache,
        acceleration=acceleration
    )
end

function estimate_outcome_mean(outcome_mean, models, dataset;
    train_validation_indices=nothing,
    cache=Dict(),
    verbosity=1,
    machine_cache=false,
    acceleration=CPU1(),
    prevalence_weights=nothing
    )
    outcome_model = acquire_model(models, outcome_mean.outcome, dataset)
    outcome_mean_estimator = ConditionalDistributionEstimator(
        outcome_model,
        train_validation_indices,
        prevalence_weights=prevalence_weights
    )
    return try_fit_ml_estimator(outcome_mean_estimator, outcome_mean, dataset;
        error_fn=outcome_mean_fit_error_msg,
        cache=cache,
        verbosity=verbosity,
        machine_cache=machine_cache,
        acceleration=acceleration
    )
end

function estimate_propensity_score_and_outcome_mean(
    acceleration::CPU1, 
    models,
    propensity_score,
    outcome_mean,
    dataset;
    train_validation_indices=nothing,
    cache=Dict(), 
    verbosity=1, 
    machine_cache=false,
    prevalence_weights=nothing
    )
    propensity_score_estimate = estimate_propensity_score(propensity_score, models, dataset;
        train_validation_indices=train_validation_indices,
        cache=cache,
        verbosity=verbosity,
        machine_cache=machine_cache,
        prevalence_weights=prevalence_weights
    )
    # Estimate outcome mean
    outcome_mean_estimate = estimate_outcome_mean(outcome_mean, models, dataset;
        train_validation_indices=train_validation_indices,
        cache=cache,
        verbosity=verbosity,
        machine_cache=machine_cache,
        prevalence_weights=prevalence_weights
    )
    return (propensity_score_estimate, outcome_mean_estimate)
end

function estimate_propensity_score_and_outcome_mean(
    acceleration::CPUThreads, 
    models,
    propensity_score,
    outcome_mean,
    dataset;
    train_validation_indices=nothing,
    cache=Dict(), 
    verbosity=1, 
    machine_cache=false,
    prevalence_weights=nothing
    )
    propensity_score_estimate = @spawn estimate_propensity_score(propensity_score, models, dataset;
        train_validation_indices=train_validation_indices,
        cache=cache,
        verbosity=verbosity,
        machine_cache=machine_cache,
        acceleration=acceleration,
        prevalence_weights=prevalence_weights

    )
    outcome_mean_estimate = @spawn estimate_outcome_mean(outcome_mean, models, dataset;
        train_validation_indices=train_validation_indices,
        cache=cache,
        verbosity=verbosity,
        machine_cache=machine_cache,
        prevalence_weights=prevalence_weights
    )
    return fetch.([propensity_score_estimate, outcome_mean_estimate])
end

estimate_censoring_score(censoring_score::Nothing, models, dataset; kwargs...) = nothing

function estimate_censoring_score(censoring_score, models, dataset;
    train_validation_indices=nothing,
    cache=Dict(),
    verbosity=1,
    machine_cache=false,
    prevalence_weights=nothing
    )
    model = acquire_model(models, censoring_score.outcome, dataset, :C_default)
    censoring_estimator = ConditionalDistributionEstimator(model, train_validation_indices, prevalence_weights=prevalence_weights)
    return try_fit_ml_estimator(censoring_estimator, censoring_score, dataset;
        error_fn=default_fit_error_msg,
        cache=cache,
        verbosity=verbosity,
        machine_cache=machine_cache
    )
end

function (estimator::CMRelevantFactorsEstimator)(estimand, dataset;
    cache=Dict(),
    verbosity=1,
    machine_cache=false,
    acceleration=CPU1()
    )
    # Lookup in cache
    estimate = estimate_from_cache(cache, estimand, estimator; verbosity=verbosity)
    estimate !== nothing && return estimate

    # Otherwise estimate
    verbosity > 0 && @info(string("Required ", string_repr(estimand)))
    models = estimator.models
    outcome_mean = estimand.outcome_mean
    propensity_score = estimand.propensity_score
    train_validation_indices = estimator.train_validation_indices
    prevalence_weights = estimator.prevalence_weights
    
    # Estimate propensity score and outcome mean
    propensity_score_estimate, outcome_mean_estimate = estimate_propensity_score_and_outcome_mean(
        acceleration,
        models,
        propensity_score,
        outcome_mean,
        dataset;
        train_validation_indices=train_validation_indices,
        cache=cache,
        verbosity=verbosity,
        machine_cache=machine_cache,
        prevalence_weights=prevalence_weights
    )

    # Estimate censoring score if needed (no prevalence_weights: censoring
    # model should not be reweighted by case-control prevalence)
    censoring_score_estimate = estimate_censoring_score(
        estimand.censoring_score, models, dataset;
        train_validation_indices=train_validation_indices,
        cache=cache,
        verbosity=verbosity,
        machine_cache=machine_cache
    )

    # Build estimate
    estimate = MLCMRelevantFactors(estimand, outcome_mean_estimate, propensity_score_estimate, censoring_score_estimate)
    # Update cache
    update_cache!(cache, estimand, estimator, estimate)
    return estimate
end

#####################################################################
###                          CMBasedTMLE                          ###
#####################################################################

struct CMBasedTMLE{T<:Union{Nothing, Tuple}}
    fluctuation::Fluctuation
    train_validation_indices::T
    prevalence_weights::Union{Nothing, Vector{Float64}}
end

CMBasedTMLE(fluctuation::Fluctuation; train_validation_indices=nothing, prevalence_weights=nothing) = 
    CMBasedTMLE(fluctuation, train_validation_indices, prevalence_weights)

function (estimator::CMBasedTMLE)(estimand, dataset; 
    cache=Dict(), 
    verbosity=1, 
    machine_cache=false,
    acceleration=CPU1()
    )
    fluctuation_model = estimator.fluctuation
    outcome_mean = fluctuation_model.initial_factors.outcome_mean.estimand

    # Fluctuate outcome model 
    fluctuated_estimator = MLConditionalDistributionEstimator(fluctuation_model, estimator.train_validation_indices)
    fluctuated_outcome_mean = try_fit_ml_estimator(fluctuated_estimator, outcome_mean, dataset;
        error_fn=outcome_mean_fluctuation_fit_error_msg,
        cache=cache,
        verbosity=verbosity,
        machine_cache=machine_cache
    )
    # Do not fluctuate propensity score or censoring score
    fluctuated_propensity_score = fluctuation_model.initial_factors.propensity_score
    censoring_score = fluctuation_model.initial_factors.censoring_score

    # Build estimate
    estimate = MLCMRelevantFactors(estimand, fluctuated_outcome_mean, fluctuated_propensity_score, censoring_score)

    return estimate
end

#####################################################################
###                       CMBasedFoldsTMLE                        ###
#####################################################################

struct CMBasedFoldsTMLE
    estimators::Vector{CMBasedTMLE}
    train_validation_indices
end

function CMBasedFoldsTMLE(Ψ, initial_factors_estimate, train_validation_indices;
    tol=nothing, 
    max_iter=1, 
    ps_lowerbound=1e-8, 
    weighted=false, 
    machine_cache=false,
    )
    estimators = map(zip(initial_factors_estimate.estimates, train_validation_indices)) do (η̂ₙ, fold_train_val_indices)
        fluctuation_model = Fluctuation(Ψ, η̂ₙ; 
            tol=tol,
            max_iter=max_iter, 
            ps_lowerbound=ps_lowerbound, 
            weighted=weighted,
            cache=machine_cache
        )
        CMBasedTMLE(fluctuation_model, train_validation_indices=fold_train_val_indices)
    end

    return CMBasedFoldsTMLE(estimators, train_validation_indices)
end


function (estimator::CMBasedFoldsTMLE)(acceleration::CPU1, estimand, dataset; 
    cache=Dict(), 
    verbosity=1, 
    machine_cache=false,
    )
    return [
        fold_estimator(estimand, dataset; 
            cache=cache, 
            verbosity=verbosity, 
            machine_cache=machine_cache
        ) for fold_estimator in estimator.estimators
    ]
end

function (estimator::CMBasedFoldsTMLE)(acceleration::CPUThreads, estimand, dataset; 
    cache=Dict(), 
    verbosity=1, 
    machine_cache=false,
    )
    n_estimators = length(estimator.estimators)
    estimates = Vector{Any}(undef, n_estimators)
    @threads for estimator_index in 1:n_estimators
        fold_estimator = estimator.estimators[estimator_index]
        estimates[estimator_index] = fold_estimator(estimand, dataset; 
            cache=cache, 
            verbosity=verbosity, 
            machine_cache=machine_cache
        )
    end
    return estimates
end

function (estimator::CMBasedFoldsTMLE)(estimand, dataset; 
    cache=Dict(), 
    verbosity=1, 
    machine_cache=false,
    acceleration=CPU1()
    )
    return estimator(acceleration, estimand, dataset; 
        cache=cache, 
        verbosity=verbosity, 
        machine_cache=machine_cache
    )
end

#####################################################################
###                           CMBasedCTMLE                        ###
#####################################################################

struct CMBasedCTMLE{S <: CollaborativeStrategy}
    fluctuation::Fluctuation
    collaborative_strategy::S
    train_validation_indices::Vector{<:Tuple}
    models::Dict
end

"""

Targeted estimator with a collaborative strategy.
"""
function (estimator::CMBasedCTMLE{S})(
    η, 
    dataset; 
    cache=Dict(), 
    verbosity=1,
    machine_cache=false,
    acceleration=CPU1()
    ) where S <: CollaborativeStrategy
    verbosity > 0 && @info "C-TMLE mode ($S)."
    collaborative_strategy = estimator.collaborative_strategy
    Ψ = estimator.fluctuation.Ψ
    fluctuation_model = estimator.fluctuation
    train_validation_indices = estimator.train_validation_indices
    
    # Retrieve models
    models = estimator.models

    # Initialize the collaborative strategy
    TMLE.initialise!(collaborative_strategy, Ψ)
    
    # Initialize Candidates: the fluctuation is fitted through the initial outcome mean and propensity score
    targeted_η̂ₙ, loss = TMLE.get_initial_candidate(η, fluctuation_model, dataset;
        verbosity=verbosity-1,
        cache=cache,
        machine_cache=machine_cache
    )

    # Initialise cross-validation loss
    cv_targeted_η̂ₙ, cv_loss = TMLE.get_initial_cv_candidate(η, dataset, fluctuation_model, train_validation_indices, models;
        cache=cache,
        verbosity=verbosity-1,
        machine_cache=machine_cache,
        acceleration=acceleration
    )

    # Collaborative Loop to find the best candidate
    candidate_info = (targeted_η̂ₙ=targeted_η̂ₙ, loss=loss, cv_targeted_η̂ₙ=cv_targeted_η̂ₙ, cv_loss=cv_loss, id=1)
    best_candidate = TMLE.find_optimal_candidate(
        candidate_info, 
        collaborative_strategy, 
        Ψ, 
        dataset, 
        fluctuation_model, 
        train_validation_indices, 
        models;
        verbosity=verbosity,
        cache=cache,
        machine_cache=machine_cache,
        acceleration=acceleration
    )
    finalise!(collaborative_strategy)

    return best_candidate.targeted_η̂ₙ
end

function get_targeted_estimator(
    Ψ, 
    collaborative_strategy, 
    train_validation_indices,
    initial_factors_estimate;
    tol=nothing,
    max_iter=1,
    ps_lowerbound=1e-8,
    weighted=true,
    machine_cache=false,
    models=nothing,
    prevalence_weights=nothing
    )
    fluctuation_model = Fluctuation(Ψ, initial_factors_estimate; 
        tol=tol,
        max_iter=max_iter, 
        ps_lowerbound=ps_lowerbound, 
        weighted=weighted,
        cache=machine_cache,
        prevalence_weights=prevalence_weights
    )
    if collaborative_strategy isa CollaborativeStrategy
        return CMBasedCTMLE(fluctuation_model, collaborative_strategy, train_validation_indices, models)
    else
        return CMBasedTMLE(fluctuation_model, train_validation_indices=nothing, prevalence_weights=prevalence_weights)
    end
end
