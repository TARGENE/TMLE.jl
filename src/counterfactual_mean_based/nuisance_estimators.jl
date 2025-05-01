#####################################################################
###             FoldsCMRelevantFactorsEstimator                   ###
#####################################################################

@auto_hash_equals struct FoldsCMRelevantFactorsEstimator <: Estimator
    models::Dict
    train_validation_indices
end

FoldsCMRelevantFactorsEstimator(models; train_validation_indices=nothing) = 
    FoldsCMRelevantFactorsEstimator(models, train_validation_indices)

function (estimator::FoldsCMRelevantFactorsEstimator)(estimand, dataset; cache=Dict(), verbosity=1, machine_cache=false)
    # Lookup in cache
    estimate = estimate_from_cache(cache, estimand, estimator; verbosity=verbosity)
    estimate !== nothing && return estimate

    # Otherwise estimate
    verbosity > 0 && @info(string("Required ", string_repr(estimand)))

    estimates = []
    for train_validation_indices in estimator.train_validation_indices
        η̂ = CMRelevantFactorsEstimator(
            train_validation_indices, 
            estimator.models
        )
        η̂ₙ = η̂(estimand, dataset; cache=cache, verbosity=verbosity, machine_cache=machine_cache)
        push!(estimates, η̂ₙ)
    end

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
end

CMRelevantFactorsEstimator(;models, train_validation_indices=nothing) = CMRelevantFactorsEstimator(train_validation_indices, models)

"""
If there is no collaborative strategy, we are in CV mode and `train_validation_indices` are used to build the initial estimator.
"""
CMRelevantFactorsEstimator(collaborative_strategy::Nothing; models, train_validation_indices=nothing) = CMRelevantFactorsEstimator(train_validation_indices, models)

"""
If there is a collaborative strategy, `train_validation_indices` are ignored to build the initial estimator.
"""
CMRelevantFactorsEstimator(collaborative_strategy; models, train_validation_indices=nothing) = CMRelevantFactorsEstimator(nothing, models)

function acquire_model(models, key, dataset, is_propensity_score)
    # If the model is in models return it
    haskey(models, key) && return models[key]
    # Otherwise, if the required model is for a propensity_score, return the default
    model_default = :G_default
    if !is_propensity_score
        # Finally, if the required model is an outcome_mean, find the type from the data
        model_default = is_binary(dataset, key) ? :Q_binary_default : :Q_continuous_default
    end
    return models[model_default]
end

function build_propensity_score_estimator(propensity_score, models, dataset;
    train_validation_indices=nothing,
    )
    cd_estimators = Dict()
    for conditional_distribution in propensity_score
        outcome = conditional_distribution.outcome
        model = acquire_model(models, outcome, dataset, true)
        cd_estimators[outcome] = ConditionalDistributionEstimator(model, train_validation_indices)
    end
    return JointConditionalDistributionEstimator(cd_estimators)
end

function estimate_propensity_score(propensity_score, models, dataset;
    train_validation_indices=nothing,
    cache=Dict(),
    verbosity=1,
    machine_cache=false
    )
    propensity_score_estimator = build_propensity_score_estimator(
        propensity_score, 
        models,  
        dataset;
        train_validation_indices=train_validation_indices,
    )
    return propensity_score_estimator(
        propensity_score, 
        dataset;
        cache=cache,
        verbosity=verbosity,
        machine_cache=machine_cache
    )
end

function estimate_outcome_mean(outcome_mean, models, dataset;
    train_validation_indices=nothing,
    cache=Dict(),
    verbosity=1,
    machine_cache=false
    )
    outcome_model = acquire_model(models, outcome_mean.outcome, dataset, false)
    outcome_mean_estimator = ConditionalDistributionEstimator(
        outcome_model,
        train_validation_indices, 
    )
    return try_fit_ml_estimator(outcome_mean_estimator, outcome_mean, dataset;
        error_fn=outcome_mean_fit_error_msg,
        cache=cache,
        verbosity=verbosity,
        machine_cache=machine_cache
    )
end

function (estimator::CMRelevantFactorsEstimator)(estimand, dataset; cache=Dict(), verbosity=1, machine_cache=false)
    # Lookup in cache
    estimate = estimate_from_cache(cache, estimand, estimator; verbosity=verbosity)
    estimate !== nothing && return estimate

    # Otherwise estimate
    verbosity > 0 && @info(string("Required ", string_repr(estimand)))
    models = estimator.models
    outcome_mean = estimand.outcome_mean
    propensity_score = estimand.propensity_score
    train_validation_indices = estimator.train_validation_indices
    # Get train validation indices
    # Estimate propensity score
    propensity_score_estimate = estimate_propensity_score(propensity_score, models, dataset;
        train_validation_indices=train_validation_indices,
        cache=cache,
        verbosity=verbosity,
        machine_cache=machine_cache
    )
    # Estimate outcome mean
    outcome_mean_estimate = estimate_outcome_mean(outcome_mean, models, dataset;
        train_validation_indices=train_validation_indices,
        cache=cache,
        verbosity=verbosity,
        machine_cache=machine_cache
    )
    # Build estimate
    estimate = MLCMRelevantFactors(estimand, outcome_mean_estimate, propensity_score_estimate)
    # Update cache
    update_cache!(cache, estimand, estimator, estimate)

    return estimate
end

#####################################################################
###              TargetedCMRelevantFactorsEstimator               ###
#####################################################################

struct TargetedCMRelevantFactorsEstimator{S <: Union{Nothing, CollaborativeStrategy}}
    fluctuation::Fluctuation
    collaborative_strategy::S
    train_validation_indices
end

function TargetedCMRelevantFactorsEstimator(Ψ, initial_factors_estimate; 
    collaborative_strategy::S=nothing,
    train_validation_indices=nothing,
    tol=nothing, 
    max_iter=1, 
    ps_lowerbound=1e-8, 
    weighted=false, 
    machine_cache=false
    ) where S <: Union{Nothing, CollaborativeStrategy}
    # If there is no collaborative strategy, we are in CV mode, we do not pass on the `train_validation_indices`
    train_validation_indices = collaborative_strategy === nothing ? nothing : train_validation_indices
    fluctuation_model = Fluctuation(Ψ, initial_factors_estimate; 
        tol=tol,
        max_iter=max_iter, 
        ps_lowerbound=ps_lowerbound, 
        weighted=weighted,
        cache=machine_cache
    )
    return TargetedCMRelevantFactorsEstimator{S}(fluctuation_model, collaborative_strategy, train_validation_indices)
end

"""

Targeted estimator in the absence of a collaborative strategy.
"""
function (estimator::TargetedCMRelevantFactorsEstimator{Nothing})(estimand, dataset; cache=Dict(), verbosity=1, machine_cache=false)
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
    # Do not fluctuate propensity score
    fluctuated_propensity_score = fluctuation_model.initial_factors.propensity_score
    # Build estimate
    estimate = MLCMRelevantFactors(estimand, fluctuated_outcome_mean, fluctuated_propensity_score)
    # Update cache
    cache[:targeted_factors] = estimate

    return estimate
end

#####################################################################
###           FoldsTargetedCMRelevantFactorsEstimator             ###
#####################################################################

struct FoldsTargetedCMRelevantFactorsEstimator
    estimators::Vector{TargetedCMRelevantFactorsEstimator}
    train_validation_indices
end

function FoldsTargetedCMRelevantFactorsEstimator(Ψ, initial_factors_estimate, train_validation_indices;
    tol=nothing, 
    max_iter=1, 
    ps_lowerbound=1e-8, 
    weighted=false, 
    machine_cache=false
    )
    estimators = map(zip(initial_factors_estimate.estimates, train_validation_indices)) do (η̂ₙ, fold_train_val_indices)
        fluctuation_model = Fluctuation(Ψ, η̂ₙ; 
            tol=tol,
            max_iter=max_iter, 
            ps_lowerbound=ps_lowerbound, 
            weighted=weighted,
            cache=machine_cache
        )
        TargetedCMRelevantFactorsEstimator(fluctuation_model, nothing, fold_train_val_indices)
    end

    return FoldsTargetedCMRelevantFactorsEstimator(estimators, train_validation_indices)
end

function (estimator::FoldsTargetedCMRelevantFactorsEstimator)(estimand, dataset; 
    cache=Dict(), 
    verbosity=1, 
    machine_cache=false
    )
    estimates = [
        fold_estimator(estimand, dataset; 
            cache=cache, 
            verbosity=verbosity, 
            machine_cache=machine_cache
        ) for fold_estimator in estimator.estimators]

    return estimates
end

#####################################################################
###  TargetedCMRelevantFactorsEstimator{<:CollaborativeStrategy}  ###
#####################################################################

"""

Targeted estimator with a collaborative strategy.
"""
function (estimator::TargetedCMRelevantFactorsEstimator{T})(
    η, 
    dataset; 
    cache=Dict(), 
    verbosity=1, 
    machine_cache=false
    ) where T <: CollaborativeStrategy
    verbosity > 0 && @info "C-TMLE mode ($T)."
    collaborative_strategy = estimator.collaborative_strategy
    Ψ = estimator.fluctuation.Ψ
    fluctuation_model = estimator.fluctuation
    train_validation_indices = estimator.train_validation_indices
    
    # Retrieve models
    models = TMLE.retrieve_models(estimator)

    # Initialize the collaborative strategy
    TMLE.initialise!(collaborative_strategy, Ψ)
    
    # Initialize Candidates: the fluctuation is fitted through the initial outcome mean and propensity score
    candidates = TMLE.initialise_candidates(η, fluctuation_model, dataset;
        verbosity=verbosity-1,
        cache=cache,
        machine_cache=machine_cache
    )

    # Initialise cross-validation loss
    cv_candidates = TMLE.initialise_cv_candidates(η, dataset, fluctuation_model, train_validation_indices, models;
        cache=cache,
        verbosity=verbosity-1,
        machine_cache=machine_cache
    )
    # Collaborative Loop to find the best candidate
    best_candidate = TMLE.update_candidates!(
        candidates, 
        cv_candidates, 
        collaborative_strategy, 
        Ψ, 
        dataset, 
        fluctuation_model, 
        train_validation_indices, 
        models;
        verbosity=verbosity,
        cache=cache,
        machine_cache=machine_cache
    )
    
    finalise!(collaborative_strategy)

    return best_candidate.candidate
end