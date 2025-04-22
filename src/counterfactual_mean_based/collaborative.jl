"""
    AdaptiveCorrelationOrdering()

This strategy can be used to adaptively select the best confounding variables for the propensity score fit. It works as follows:

1. The propensity score is fitted with no confounding variables
2. Until convergence (or all confounding variables have been added): for each remaining confounding variable, a new propensity score is trained and an associated targeted estiamtor is built. The estimator with the lowest error is selected.
3. The sequence of models is evaluated via penalized cross-validation.
"""
struct AdaptiveCorrelationOrdering <: CollaborativeStrategy 
    resampling::ResamplingStrategy
end

AdaptiveCorrelationOrdering(;resampling=StratifiedCV()) = AdaptiveCorrelationOrdering(resampling)

#####################################################################
###           AdaptiveCorrelationOrderingPSEstimator              ###
#####################################################################

struct AdaptiveCorrelationOrderingPSEstimator <: Estimator
    cd_estimators::Dict
    confounders::Vector{Symbol}
end

CollaborativePSEstimator(collaborative_strategy::AdaptiveCorrelationOrdering, cd_estimators; confounders=Symbol[]) =
    AdaptiveCorrelationOrderingPSEstimator(cd_estimators, confounders)

function (estimator::AdaptiveCorrelationOrderingPSEstimator)(conditional_distributions, dataset; cache=Dict(), verbosity=1, machine_cache=false)
    # Add Intercept to dataset
    dataset = merge(dataset, (;COLLABORATIVE_INTERCEPT=ones(nrows(dataset))))

    # Define the restricted conditional distributions
    treatment_variables = outcome_set(conditional_distributions)

    restricted_conditional_distributions = map(conditional_distributions) do conditional_distribution
        # Only keep parents in the restricted set or being a treatment variable
        new_parents = filter(x -> x ∈ union(estimator.confounders, treatment_variables), conditional_distribution.parents)
        # Add the intercept
        new_parents = (new_parents..., :COLLABORATIVE_INTERCEPT)
        ConditionalDistribution(conditional_distribution.outcome, new_parents)
    end

    estimates = fit_conditional_distributions(estimator.cd_estimators, restricted_conditional_distributions, dataset; 
        cache=cache, 
        verbosity=verbosity, 
        machine_cache=machine_cache
    )
    return AdaptiveCorrelationOrderingPSEstimate(restricted_conditional_distributions, estimates)
end

#####################################################################
###           AdaptiveCorrelationOrderingPSEstimate               ###
#####################################################################

struct AdaptiveCorrelationOrderingPSEstimate{T, N} <: Estimate
    estimand::Tuple{Vararg{ConditionalDistribution, N}}
    components::Tuple{Vararg{T, N}}
end

#####################################################################
###              Targeting AdaptiveCorrelationOrdering            ###
#####################################################################

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
    dataset = merge(dataset, (;COLLABORATIVE_INTERCEPT=ones(nrows(dataset))))
    fluctuation_model = estimator.fluctuation
    outcome = η.outcome_mean.outcome
    outcome_mean_estimate = estimator.fluctuation.initial_factors.outcome_mean
    propensity_score_estimate = estimator.fluctuation.initial_factors.propensity_score
    propensity_score_models = Dict(cd.outcome => cde.machine.model for (cd, cde) in zip(propensity_score_estimate.estimand, propensity_score_estimate.components))
    treatment_variables = TMLE.outcome_set(η.propensity_score)
    remaining_confounders = TMLE.get_confounders(η.propensity_score, treatment_variables)
    # Initialize Candidates: the fluctuation is fitted through the initial outcome mean and propensity score
    candidate, loss = TMLE.get_new_targeted_candidate(η, fluctuation_model, dataset;
        verbosity=verbosity-1,
        machine_cache=machine_cache
        )
    candidates = [(estimates=candidate, loss=loss)]
    # Loop through variables
    current_confounders = Symbol[]
    while length(remaining_confounders) > 0
        # Find the confounder most correlated with the residuals
        residuals_ = TMLE.compute_residuals(outcome_mean_estimate, outcome, dataset)
        TMLE.update_with_most_correlated!(remaining_confounders, current_confounders, dataset, residuals_)
        verbosity > 0 && @info "The propensity score will use: $(current_confounders)"
        # Fit propensity score
        propensity_score_estimator = TMLE.build_propensity_score_estimator(
            η.propensity_score,
            propensity_score_models,  
            dataset;
            collaborative_strategy=estimator.collaborative_strategy, 
            train_validation_indices=nothing,
            confounders=current_confounders
        )
        propensity_score_estimate = propensity_score_estimator(
            η.propensity_score, 
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
                outcome_mean_estimate=last(candidates).estimates.outcome_mean,
                verbosity=verbosity-1,
                machine_cache=machine_cache
            )
        end
        push!(candidates, (estimates=candidate, loss=loss))
    end

    y = Tables.getcolumn(dataset, outcome)
    for (train, val) in MLJBase.train_test_pairs(estimator.collaborative_strategy.resampling, 1:nrows(dataset), y)
        for candidate in candidates
            
        end
    end
    # Select the best candidate by penalized cross-validation
    
    return candidates
end

#####################################################################
###                           Functions                           ###
#####################################################################

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

function update_with_most_correlated!(remaining_confounders, current_confounders, dataset, residuals)
    max_cor = 0.
    best_confounder = :nothing
    best_index = 0
    for (index, confounder) in enumerate(remaining_confounders)
        σ = abs(cor(Tables.getcolumn(dataset, confounder), residuals))
        if σ > max_cor
            max_cor = σ
            best_confounder = confounder
            best_index = index
        end
    end
    popat!(remaining_confounders, best_index)
    push!(current_confounders, best_confounder)
end