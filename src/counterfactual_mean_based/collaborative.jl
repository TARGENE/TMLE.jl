"""
    AdaptiveCorrelationOrdering()

This strategy can be used to adaptively select the best confounding variables for the propensity score fit. It works as follows:

1. The propensity score is fitted with no confounding variables
2. Until convergence (or all confounding variables have been added): for each remaining confounding variable, a new propensity score is trained and an associated targeted estiamtor is built. The estimator with the lowest error is selected.
3. The sequence of models is evaluated via penalized cross-validation.
"""
struct AdaptiveCorrelationOrdering <: CollaborativeStrategy 
    resampling::ResamplingStrategy
    remaining_confounders::Set{Symbol}
    current_confounders::Set{Symbol}
    AdaptiveCorrelationOrdering(resampling) = new(resampling, Set{Symbol}(), Set{Symbol}())
end

AdaptiveCorrelationOrdering(;resampling=StratifiedCV()) = AdaptiveCorrelationOrdering(resampling)

function initialise!(strategy::AdaptiveCorrelationOrdering, Ψ)
    empty!(strategy.remaining_confounders)
    union!(strategy.remaining_confounders, Set(Iterators.flatten(values(Ψ.treatment_confounders))))
    empty!(strategy.current_confounders)
    return nothing
end

function update!(strategy, dataset, residuals)
    max_cor = 0.
    best_confounder = :nothing
    for confounder in strategy.remaining_confounders
        σ = abs(cor(Tables.getcolumn(dataset, confounder), residuals))
        if σ > max_cor
            max_cor = σ
            best_confounder = confounder
        end
    end
    delete!(strategy.remaining_confounders, best_confounder)
    push!(strategy.current_confounders, best_confounder)
    return nothing
end


function finalise!(strategy::AdaptiveCorrelationOrdering)
    empty!(strategy.remaining_confounders)
    empty!(strategy.current_confounders)
    return nothing
end

function propensity_score(Ψ::StatisticalCMCompositeEstimand, collaborative_strategy::AdaptiveCorrelationOrdering)
    Ψtreatments = TMLE.treatments(Ψ)
    return Tuple(map(eachindex(Ψtreatments)) do index
        T = Ψtreatments[index]
        confounders = (Ψtreatments[index+1:end]..., collaborative_strategy.current_confounders..., :COLLABORATIVE_INTERCEPT)
        ConditionalDistribution(T, confounders)
    end)
end

#####################################################################
###              Targeting AdaptiveCorrelationOrdering            ###
#####################################################################

function retrieve_propensity_score_models(estimator)
    propensity_score_estimate = estimator.fluctuation.initial_factors.propensity_score
    return Dict(cd.outcome => cde.machine.model for (cd, cde) in zip(propensity_score_estimate.estimand, propensity_score_estimate.components))
end

function get_new_propensity_score(collaborative_strategy::AdaptiveCorrelationOrdering, Ψ, last_candidate, dataset)
    Q̂n = last_candidate.outcome_mean
    residuals = compute_residuals(Q̂n, Ψ.outcome, dataset)
    update!(collaborative_strategy, dataset, residuals)
    return propensity_score(Ψ, collaborative_strategy)
end

function get_new_targeted_candidate(η, fluctuation_model, dataset;
    propensity_score_estimate=fluctuation_model.initial_factors.propensity_score,
    outcome_mean_estimate=fluctuation_model.initial_factors.outcome_mean,
    verbosity=1,
    machine_cache=false
    )
    fluctuation_model = update_fluctuation_model(fluctuation_model; 
        propensity_score_estimate=propensity_score_estimate,
        outcome_mean_estimate=outcome_mean_estimate
    )
    fluctuated_estimator = MLConditionalDistributionEstimator(fluctuation_model)
    fluctuated_outcome_mean = try_fit_ml_estimator(fluctuated_estimator, η.outcome_mean, dataset;
        error_fn=outcome_mean_fluctuation_fit_error_msg,
        verbosity=verbosity,
        machine_cache=machine_cache
    )
    candidate = MLCMRelevantFactors(η, fluctuated_outcome_mean, propensity_score_estimate)
    loss = evaluate_candidate(candidate, η.outcome_mean.outcome, dataset)
    return candidate, loss
end

"""

Targeted estimator with a collaborative strategy.
"""
function (estimator::TargetedCMRelevantFactorsEstimator{AdaptiveCorrelationOrdering})(
    η, 
    dataset; 
    cache=Dict(), 
    verbosity=1, 
    machine_cache=false
    )
    collaborative_strategy = estimator.collaborative_strategy
    Ψ = estimator.fluctuation.Ψ
    fluctuation_model = estimator.fluctuation
    propensity_score_models = retrieve_propensity_score_models(estimator)
    # Initialize the collaborative strategy
    TMLE.initialise!(estimator.collaborative_strategy, Ψ)
    
    # Initialize Candidates: the fluctuation is fitted through the initial outcome mean and propensity score
    candidate, loss = TMLE.get_new_targeted_candidate(η, fluctuation_model, dataset;
        verbosity=verbosity-1,
        machine_cache=machine_cache
        )
    candidates = [(estimates=candidate, loss=loss)]
    # Initialise cross-validation loss
    # η̂_cv = CMRelevantFactorsEstimator(estimator.collaborative_strategy.resampling, estimator.models)

    # Collaborative Loop
    while length(collaborative_strategy.remaining_confounders) > 0
        # Find the confounder most correlated with the residuals
        last_candidate = last(candidates).estimates
        propensity_score = TMLE.get_new_propensity_score(collaborative_strategy, Ψ, last_candidate, dataset)
        verbosity > 0 && @info "The propensity score will use: $(propensity_score)"
        # Estimate propensity score on full dataset
        propensity_score_estimator = build_propensity_score_estimator(
            propensity_score, 
            propensity_score_models,  
            dataset;
            train_validation_indices=nothing,
        )
        propensity_score_estimate = propensity_score_estimator(
            propensity_score, 
            dataset;
            cache=cache,
            verbosity=verbosity-1,
            machine_cache=machine_cache
        )
        # Fluctuate outcome model through the new propensity score
        candidate, loss = get_new_targeted_candidate(η, fluctuation_model, dataset;
            propensity_score_estimate=propensity_score_estimate,
            verbosity=verbosity-1,
            machine_cache=machine_cache
        )
        if loss > last(candidates).loss
            # Fluctuate through Q̄* from the previous candidate
            candidate, loss = get_new_targeted_candidate(η, fluctuation_model, dataset;
                propensity_score_estimate=propensity_score_estimate,
                outcome_mean_estimate=last_candidate.outcome_mean,
                verbosity=verbosity-1,
                machine_cache=machine_cache
            )
        end

        push!(candidates, (estimates=candidate, loss=loss))
    end
    # Select the best candidate by penalized cross-validation
    
    return candidates
end

#####################################################################
###                           Functions                           ###
#####################################################################

function cv_evaluate_candidate(candidate, propensity_score_estimator, η, dataset; 
    resampling=StratifiedCV(),
    cache=Dict(),
    verbosity=1,
    machine_cache=false
    )
    outcome = η.outcome_mean.outcome
    y = Tables.getcolumn(dataset, outcome)
    for (train, val) in MLJBase.train_test_pairs(resampling, 1:nrows(dataset), dataset, y)
        dataset_train = selectrows(dataset, train)
        dataset_val = selectrows(dataset, val)
        propensity_score_estimate = propensity_score_estimator(
            η.propensity_score, 
            dataset_train;
            cache=cache,
            verbosity=verbosity-1,
            machine_cache=machine_cache
        )
        candidate, loss = get_new_targeted_candidate(η, fluctuation_model, dataset_train;
            propensity_score_estimate=propensity_score_estimate,
            outcome_mean_estimate=#TODO,
            verbosity=verbosity-1,
            machine_cache=machine_cache
        )
    end
end
evaluate_candidate(candidate, outcome, dataset) =
    mean(TMLE.compute_residuals(candidate.outcome_mean, outcome, dataset))

function update_fluctuation_model(fluctuation_model;
    propensity_score_estimate=fluctuation_model.initial_factors.propensity_score, 
    outcome_mean_estimate=fluctuation_model.initial_factors.outcome_mean
    )
    η̂ = TMLE.MLCMRelevantFactors(
        fluctuation_model.initial_factors.estimand,
        outcome_mean_estimate,
        propensity_score_estimate
    )
    return Fluctuation(fluctuation_model.Ψ, η̂; 
        tol=fluctuation_model.tol,
        max_iter=fluctuation_model.max_iter, 
        ps_lowerbound=fluctuation_model.ps_lowerbound, 
        weighted=fluctuation_model.weighted,
        cache=fluctuation_model.cache
    )
end

adapt_and_getloss(ŷ::Vector{<:Real}) = ŷ, RootMeanSquaredError()
adapt_and_getloss(ŷ::Vector{<:Distribution{Univariate, Distributions.Continuous}}) = TMLE.expected_value(ŷ), RootMeanSquaredError()

function compute_residuals(conditional_density_estimate, outcome, dataset)
    ŷ = MLJBase.predict(conditional_density_estimate, dataset)
    y = Tables.getcolumn(dataset, outcome)
    ŷ, loss = TMLE.adapt_and_getloss(ŷ)
    return measurements(loss, ŷ, y)
end

outcome_set(conditional_distributions) = Set(cd.outcome for cd in conditional_distributions)

get_confounders(conditional_distributions, treatments) = setdiff(
    union((cd.parents for cd in conditional_distributions)...),
    treatments
)