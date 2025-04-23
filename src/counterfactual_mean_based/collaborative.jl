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

function retrieve_models(estimator)
    outcome_mean_estimate = estimator.fluctuation.initial_factors.outcome_mean
    propensity_score_estimate = estimator.fluctuation.initial_factors.propensity_score
    models = Dict{Symbol, Any}(outcome_mean_estimate.estimand.outcome => outcome_mean_estimate.machine.model)
    for (cd, cde) in zip(propensity_score_estimate.estimand, propensity_score_estimate.components)
        models[cd.outcome] = cde.machine.model 
    end
    return models
end

function get_new_propensity_score(collaborative_strategy::AdaptiveCorrelationOrdering, Ψ, last_candidate, dataset)
    Q̂n = last_candidate.outcome_mean
    residuals = compute_residuals(Q̂n, dataset)
    update!(collaborative_strategy, dataset, residuals)
    return propensity_score(Ψ, collaborative_strategy)
end


function initialise_candidates(η, targeted_η̂_template, dataset;
    verbosity=1,
    cache=Dict(),
    machine_cache=false
    )
    targeted_η̂ = TMLE.TargetedCMRelevantFactorsEstimator(
        targeted_η̂_template.fluctuation, 
        nothing
    )
    targeted_η̂ₙ = targeted_η̂(η, dataset;
        cache=cache,
        verbosity=verbosity,
        machine_cache=machine_cache
    )
    loss = evaluate_candidate(targeted_η̂ₙ, dataset)
    return [(estimates=targeted_η̂ₙ, loss=loss)]
end


function get_new_targeted_candidate(η, last_candidate, targeted_η̂_template, propensity_score_estimate, dataset;
    use_fluct=false,
    verbosity=1,
    cache=Dict(),
    machine_cache=false
    )
    # Define new nuisance factors estimate
    Q̄ₙ = use_fluct ? last_candidate.outcome_mean : last_candidate.outcome_mean.machine.model.initial_factors.outcome_mean
    η̂ₙ = TMLE.MLCMRelevantFactors(η, Q̄ₙ, propensity_score_estimate)
    # Fluctuate
    targeted_η̂ = TMLE.TargetedCMRelevantFactorsEstimator(
            targeted_η̂_template.fluctuation.Ψ, 
            η̂ₙ;
            tol=targeted_η̂_template.fluctuation.tol,
            max_iter=targeted_η̂_template.fluctuation.max_iter,
            ps_lowerbound=targeted_η̂_template.fluctuation.ps_lowerbound,
            weighted=targeted_η̂_template.fluctuation.weighted,
            machine_cache=targeted_η̂_template.fluctuation.cache
        )
    targeted_η̂ₙ = targeted_η̂(η, dataset;
        cache=cache,
        verbosity=verbosity-1,
        machine_cache=machine_cache
    )
    loss = evaluate_candidate(targeted_η̂ₙ, dataset)
    return targeted_η̂ₙ, loss
end


function get_new_propensity_score_and_estimator(
    collaborative_strategy::AdaptiveCorrelationOrdering, 
    Ψ, 
    last_candidate, 
    dataset,
    models)
    propensity_score = TMLE.get_new_propensity_score(collaborative_strategy, Ψ, last_candidate, dataset)
    propensity_score_estimator = TMLE.build_propensity_score_estimator(
        propensity_score, 
        models,  
        dataset;
        train_validation_indices=nothing,
    )
    return propensity_score, propensity_score_estimator
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
    
    # Retrieve models
    models = TMLE.retrieve_models(estimator)

    # Initialize the collaborative strategy
    TMLE.initialise!(estimator.collaborative_strategy, Ψ)
    
    # Initialize Candidates: the fluctuation is fitted through the initial outcome mean and propensity score
    candidates = TMLE.initialise_candidates(η, estimator, dataset;
        verbosity=verbosity,
        cache=cache,
        machine_cache=machine_cache
    )

    # Initialise cross-validation loss
    cv_candidates = TMLE.initialise_cv_candidates(η, dataset, estimator, models;
        cache=Dict(),
        verbosity=verbosity-1,
        machine_cache=machine_cache
    )

    # Collaborative Loop
    while length(collaborative_strategy.remaining_confounders) > 0
        # Find the confounder most correlated with the residuals
        last_candidate = last(candidates).estimates
        propensity_score, propensity_score_estimator = TMLE.get_new_propensity_score_and_estimator(
            collaborative_strategy, 
            Ψ, 
            last_candidate, 
            dataset,
            models
        )
        verbosity > 0 && @info "The propensity score will use: $(propensity_score)"
        propensity_score_estimate = propensity_score_estimator(
            propensity_score,
            dataset;
            cache=cache,
            verbosity=verbosity-1,
            machine_cache=machine_cache
        )
        # Fluctuate outcome model through the new propensity score
        new_η = TMLE.CMRelevantFactors(η.outcome_mean, propensity_score)
        use_fluct = false
        candidate, loss = TMLE.get_new_targeted_candidate(new_η, last_candidate, estimator, propensity_score_estimate, dataset;
            use_fluct=use_fluct,
            verbosity=verbosity,
            cache=cache,
            machine_cache=machine_cache
        )
        if loss > last(candidates).loss
            use_fluct = true
            # Fluctuate through Q̄* from the previous candidate's flutuated model
            candidate, loss = TMLE.get_new_targeted_candidate(new_η, last_candidate, estimator, propensity_score_estimate, dataset;
                use_fluct=true,
                verbosity=verbosity,
                cache=cache,
                machine_cache=machine_cache
            )
        end
        push!(candidates, (estimates=candidate, loss=loss))
        # Evaluate candidate
        TMLE.evaluate_cv_candidate!(cv_candidates, new_η, targeted_η̂_template, propensity_score, propensity_score_estimator, dataset; 
            use_fluct=use_fluct,
            verbosity=verbosity,
            cache=Dict(),
            machine_cache=machine_cache
        )
    end
    # Select the best candidate by penalized cross-validation
    
    return candidates
end

#####################################################################
###                           Functions                           ###
#####################################################################

function initialise_cv_candidates(η, dataset, targeted_η̂_template, models;
    cache=Dict(),
    verbosity=1,
    machine_cache=false
    )
    resampling = targeted_η̂_template.collaborative_strategy.resampling
    outcome = η.outcome_mean.outcome
    y = Tables.getcolumn(dataset, outcome)
    val_loss = 0.
    η̂ = TMLE.CMRelevantFactorsEstimator(;models=models)
    cv_candidate = []
    for (train, val) in MLJBase.train_test_pairs(resampling, 1:nrows(dataset), dataset, y)
        # Split dataset into training and validation sets
        dataset_train = selectrows(dataset, train)
        dataset_val = selectrows(dataset, val)
        # Estimate initial factors
        η̂ₙ_train = η̂(η, dataset_train;
            cache=cache,
            verbosity=verbosity-1,
            machine_cache=machine_cache
        )
        # Target Q̄ estimator
        targeted_η̂_train = TMLE.TargetedCMRelevantFactorsEstimator(
            targeted_η̂_template.fluctuation.Ψ, 
            η̂ₙ_train;
            collaborative_strategy=nothing,
            tol=targeted_η̂_template.fluctuation.tol,
            max_iter=targeted_η̂_template.fluctuation.max_iter,
            ps_lowerbound=targeted_η̂_template.fluctuation.ps_lowerbound,
            weighted=targeted_η̂_template.fluctuation.weighted,
            machine_cache=machine_cache
        )
        targeted_η̂ₙ_train = targeted_η̂_train(η, dataset_train;
            cache=cache,
            verbosity=verbosity-1,
            machine_cache=machine_cache
        )
        val_loss += evaluate_candidate(targeted_η̂ₙ_train, dataset_val)
        push!(cv_candidate, targeted_η̂ₙ_train)
    end
    return [(estimates=cv_candidate, loss=val_loss)]
end


function evaluate_cv_candidate!(cv_candidates, η, targeted_η̂_template, propensity_score, propensity_score_estimator, dataset; 
    use_fluct=false,
    verbosity=1,
    cache=Dict(),
    machine_cache=false
    )
    resampling = targeted_η̂_template.collaborative_strategy.resampling
    last_cv_candidate = last(cv_candidates).estimates
    outcome = η.outcome_mean.outcome
    y = Tables.getcolumn(dataset, outcome)
    val_loss = 0.
    for (index, (train, val)) in enumerate(MLJBase.train_test_pairs(resampling, 1:nrows(dataset), dataset, y))
        # Split dataset into training and validation sets
        dataset_train = selectrows(dataset, train)
        dataset_val = selectrows(dataset, val)
        # Update propensity score estimate
        propensity_score_estimate = propensity_score_estimator(
            propensity_score,
            dataset_train;
            cache=cache,
            verbosity=verbosity,
            machine_cache=machine_cache
        )
        # Target Q̄ estimator
        targeted_η̂ₙ_train, _ = get_new_targeted_candidate(η, last_cv_candidate[index], targeted_η̂_template, propensity_score_estimate, dataset;
            use_fluct=use_fluct,
            verbosity=verbosity,
            cache=cache,
            machine_cache=machine_cache
        )
        val_loss += evaluate_candidate(targeted_η̂ₙ_train, dataset_val)
    end
    push!(cv_candidates, (estimates=cv_candidates, loss=val_loss))
    return nothing
end

evaluate_candidate(candidate, dataset) =
    sum(TMLE.compute_residuals(candidate.outcome_mean, dataset))

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

function compute_residuals(conditional_density_estimate, dataset)
    ŷ = MLJBase.predict(conditional_density_estimate, dataset)
    y = Tables.getcolumn(dataset, conditional_density_estimate.estimand.outcome)
    ŷ, loss = TMLE.adapt_and_getloss(ŷ)
    return measurements(loss, ŷ, y)
end

outcome_set(conditional_distributions) = Set(cd.outcome for cd in conditional_distributions)

get_confounders(conditional_distributions, treatments) = setdiff(
    union((cd.parents for cd in conditional_distributions)...),
    treatments
)