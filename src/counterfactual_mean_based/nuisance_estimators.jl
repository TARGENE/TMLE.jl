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
            train_validation_indices, 
            estimator.models
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
            train_validation_indices, 
            estimator.models
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
    prevalence::Union{Nothing, Float64}
end

CMRelevantFactorsEstimator(;models, train_validation_indices=nothing, prevalence=nothing) = CMRelevantFactorsEstimator(train_validation_indices, models, prevalence)
"""
Option to maintain compatibility with the old API.
"""
CMRelevantFactorsEstimator(train_validation_indices, models; prevalence=nothing) = CMRelevantFactorsEstimator(train_validation_indices, models, prevalence)
"""
If there is no collaborative strategy, we are in CV mode and `train_validation_indices` are used to build the initial estimator.
"""
CMRelevantFactorsEstimator(collaborative_strategy::Nothing; models, train_validation_indices=nothing, prevalence=nothing) = CMRelevantFactorsEstimator(train_validation_indices, models, prevalence)

"""
If there is a collaborative strategy, `train_validation_indices` are ignored to build the initial estimator.
"""
CMRelevantFactorsEstimator(collaborative_strategy; models, train_validation_indices=nothing, prevalence=nothing) = CMRelevantFactorsEstimator(nothing, models, prevalence)

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
    prevalence_weights=nothing
    )
    cd_estimators = Dict()
    for conditional_distribution in propensity_score
        outcome = conditional_distribution.outcome
        model = acquire_model(models, outcome, dataset, true)
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
    outcome_model = acquire_model(models, outcome_mean.outcome, dataset, false)
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
    # Bug here for CMBasedCTMLE prevalence_weights are generated from the whole data
    prevalence_weights = get_weights_from_prevalence(estimator.prevalence, dataset[!, outcome_mean.outcome])
    
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
    # If the prevalence is provided, estimate the marginal distribution of the covariates
    if estimand.marginal_w !== nothing 
        marginals = []
        for w in estimand.marginal_w
            marg_est = MarginalDistributionEstimator(
                w.variable,
                train_validation_indices;
                prevalence_weights = prevalence_weights
            )
            push!(marginals, marg_est(
                w,
                dataset;
                cache         = cache,
                verbosity     = verbosity,
                machine_cache = machine_cache,
                acceleration  = acceleration
            ))
        end
    else
        marginals = nothing
    end
    
    # Build estimate
    estimate = MLCMRelevantFactors(estimand, outcome_mean_estimate, propensity_score_estimate, marginals)
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

CMBasedTMLE(fluctuation::Fluctuation, train_validation_indices; prevalence_weights=nothing) = 
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
    fluctuated_estimator = MLConditionalDistributionEstimator(fluctuation_model, estimator.train_validation_indices, estimator.prevalence_weights)
    fluctuated_outcome_mean = try_fit_ml_estimator(fluctuated_estimator, outcome_mean, dataset;
        error_fn=outcome_mean_fluctuation_fit_error_msg,
        cache=cache,
        verbosity=verbosity,
        machine_cache=machine_cache
    )
    # Do not fluctuate propensity score
    fluctuated_propensity_score = fluctuation_model.initial_factors.propensity_score
    # Do not fluctuate marginal distribution of W 
    marginal_w = fluctuation_model.initial_factors.marginal_w
    # Build estimate
    estimate = MLCMRelevantFactors(estimand, fluctuated_outcome_mean, fluctuated_propensity_score, marginal_w)

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
        CMBasedTMLE(fluctuation_model, fold_train_val_indices)
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
    verbosity > 0 && @info "Initializing collaborative strategy."
    TMLE.initialise!(collaborative_strategy, Ψ)
    
    # Initialize Candidates: the fluctuation is fitted through the initial outcome mean and propensity score
    verbosity > 0 && @info "Initializing candidates."
    targeted_η̂ₙ, loss = TMLE.get_initial_candidate(η, fluctuation_model, dataset;
        verbosity=verbosity-1,
        cache=cache,
        machine_cache=machine_cache
    )

    # Initialise cross-validation loss
    verbosity > 0 && @info "Initializing CV loss."
    cv_targeted_η̂ₙ, cv_loss = TMLE.get_initial_cv_candidate(η, dataset, fluctuation_model, train_validation_indices, models;
        cache=cache,
        verbosity=verbosity-1,
        machine_cache=machine_cache,
        acceleration=acceleration
    )

    # Collaborative Loop to find the best candidate
    verbosity > 0 && @info "Finding optimal candidate."
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
    verbosity > 0 && @info "Finalizing."
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
