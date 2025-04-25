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

function update!(strategy, last_candidate, dataset)
    residuals = compute_loss(last_candidate.outcome_mean, dataset)
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

function initialise_candidates(η, fluctuation_model, dataset;
    verbosity=1,
    cache=Dict(),
    machine_cache=false
    )
    targeted_η̂ = TMLE.TargetedCMRelevantFactorsEstimator(
        fluctuation_model, 
        nothing,
        nothing
    )
    targeted_η̂ₙ = targeted_η̂(η, dataset;
        cache=cache,
        verbosity=verbosity,
        machine_cache=machine_cache
    )
    loss = loss_sum(targeted_η̂ₙ, dataset)
    return [(candidate=targeted_η̂ₙ, loss=loss)]
end


function get_new_targeted_candidate(last_candidate, new_propensity_score_estimate, fluctuation_model, dataset;
    use_fluct=false,
    verbosity=1,
    cache=Dict(),
    machine_cache=false
    )
    new_η = TMLE.CMRelevantFactors(last_candidate.estimand.outcome_mean, new_propensity_score_estimate.estimand)
    # Define new nuisance factors estimate
    η̂ₙ = TMLE.MLCMRelevantFactors(
        new_η, 
        use_fluct ? last_candidate.outcome_mean : last_candidate.outcome_mean.machine.model.initial_factors.outcome_mean, 
        new_propensity_score_estimate
    )
    # Fluctuate
    targeted_η̂ = TMLE.TargetedCMRelevantFactorsEstimator(
            fluctuation_model.Ψ, 
            η̂ₙ;
            tol=fluctuation_model.tol,
            max_iter=fluctuation_model.max_iter,
            ps_lowerbound=fluctuation_model.ps_lowerbound,
            weighted=fluctuation_model.weighted,
            machine_cache=fluctuation_model.cache
        )
    targeted_η̂ₙ = targeted_η̂(new_η, dataset;
        cache=cache,
        verbosity=verbosity,
        machine_cache=machine_cache
    )
    loss = loss_sum(targeted_η̂ₙ, dataset)
    return targeted_η̂ₙ, loss
end

function get_new_propensity_score_and_estimator(
    collaborative_strategy, 
    Ψ, 
    dataset,
    models
    )
    new_propensity_score = propensity_score(Ψ, collaborative_strategy)
    new_propensity_score_estimator = build_propensity_score_estimator(
        new_propensity_score, 
        models,  
        dataset;
        train_validation_indices=nothing,
    )
    return new_propensity_score, new_propensity_score_estimator
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
    train_validation_indices = MLJBase.train_test_pairs(
        estimator.collaborative_strategy.resampling, 
        1:nrows(dataset), 
        dataset, 
        Tables.getcolumn(dataset, η.outcome_mean.outcome)
    )
    
    # Retrieve models
    models = TMLE.retrieve_models(estimator)

    # Initialize the collaborative strategy
    TMLE.initialise!(estimator.collaborative_strategy, Ψ)
    
    # Initialize Candidates: the fluctuation is fitted through the initial outcome mean and propensity score
    candidates = TMLE.initialise_candidates(η, fluctuation_model, dataset;
        verbosity=verbosity,
        cache=cache,
        machine_cache=machine_cache
    )

    # Initialise cross-validation loss
    cv_candidates = TMLE.initialise_cv_candidates(η, dataset, fluctuation_model, train_validation_indices, models;
        cache=cache,
        verbosity=verbosity,
        machine_cache=machine_cache
    )

    # Collaborative Loop to find the best candidate
    candidate_id = 1
    best_candidate = (candidate=only(candidates).candidate, cvloss=only(cv_candidates).loss, id=candidate_id)
    while length(collaborative_strategy.remaining_confounders) > 0
        candidate_id += 1
        # Update the collaborative strategy's state
        last_candidate, last_candidate_loss = last(candidates)
        update!(strategy, last_candidate, dataset)

        verbosity > 0 && @info "The propensity score will use: $(propensity_score)"
        new_propensity_score, new_propensity_score_estimator = TMLE.get_new_propensity_score_and_estimator(
            collaborative_strategy, 
            Ψ, 
            dataset,
            models
        )
        new_propensity_score_estimate = new_propensity_score_estimator(
            new_propensity_score,
            dataset;
            cache=cache,
            verbosity=verbosity,
            machine_cache=machine_cache
        )
        # Fluctuate outcome model through the new propensity score
        use_fluct = false
        candidate, loss = TMLE.get_new_targeted_candidate(last_candidate, new_propensity_score_estimate, fluctuation_model, dataset;
            use_fluct=use_fluct,
            verbosity=verbosity,
            cache=cache,
            machine_cache=machine_cache
        )
        if loss > last(candidates).loss
            use_fluct = true
            # Fluctuate through Q̄* from the previous candidate's flutuated model
            candidate, loss = TMLE.get_new_targeted_candidate(last_candidate, new_propensity_score_estimate, fluctuation_model, dataset;
                use_fluct=use_fluct,
                verbosity=verbosity,
                cache=cache,
                machine_cache=machine_cache
            )
        end
        push!(candidates, (candidate=candidate, loss=loss))
        # Evaluate candidate
        last_cv_candidate, last_cv_loss = last(cv_candidates)
        cv_candidate, cv_loss = TMLE.evaluate_cv_candidate!(last_cv_candidate, fluctuation_model, new_propensity_score, models, dataset, train_validation_indices; 
            use_fluct=use_fluct,
            verbosity=verbosity,
            cache=cache,
            machine_cache=machine_cache
        )
        push!(cv_candidates, (candidate=cv_candidate, loss=cv_loss))
        # Update the best candidate or early stop
        if cv_loss < last_cv_loss
            best_candidate = (candidate=candidate, loss=cv_loss, id=candidate_id)
        elseif candidate_id - best_candidate.id > estimator.patience
            break
        end
    end
    
    finalise!(collaborative_strategy)

    return best_candidate.candidate
end

#####################################################################
###                           Functions                           ###
#####################################################################

function initialise_cv_candidates(η, dataset, fluctuation_model, train_validation_indices, models;
    cache=Dict(),
    verbosity=1,
    machine_cache=false
    )
    # Estimate Nuisance parameters on each fold
    η̂ = TMLE.FoldsCMRelevantFactorsEstimator(models, train_validation_indices)
    η̂ₙ = η̂(η, dataset;
        cache=cache,
        verbosity=verbosity,
        machine_cache=machine_cache
    )
    # Target Nuisance parameters on each fold
    targeted_η̂ = TMLE.FoldsTargetedCMRelevantFactorsEstimator(
            fluctuation_model.Ψ, 
            η̂ₙ,
            train_validation_indices;
            tol=fluctuation_model.tol,
            max_iter=fluctuation_model.max_iter,
            ps_lowerbound=fluctuation_model.ps_lowerbound,
            weighted=fluctuation_model.weighted,
            machine_cache=machine_cache
        )
    candidate = targeted_η̂(η, dataset;
            cache=cache,
            verbosity=verbosity,
            machine_cache=machine_cache
        )
    # Evaluate candidate on validation fold
    validation_loss = compute_validation_loss(candidate, dataset, train_validation_indices)

    return [(candidate=candidate, loss=validation_loss)]
end

function compute_validation_loss(candidate, dataset, train_validation_indices)
    return mapreduce(+, zip(train_validation_indices, candidate)) do ((_, val_indices), targeted_η̂ₙ)
        validation_dataset = selectrows(dataset, val_indices)
        loss_sum(targeted_η̂ₙ, validation_dataset)
    end
end

function evaluate_cv_candidate!(last_cv_candidate, fluctuation_model, propensity_score, models, dataset, train_validation_indices; 
    use_fluct=false,
    verbosity=1,
    cache=Dict(),
    machine_cache=false
    )
    validation_loss = 0.
    fold_candidates = []
    for (fold_index, fold_train_val_indices) in enumerate(train_validation_indices)
        # Update propensity score estimate
        fold_propensity_score_estimator = build_propensity_score_estimator(
            propensity_score, 
            models,  
            dataset;
            train_validation_indices=fold_train_val_indices,
        )
        fold_propensity_score_estimate = fold_propensity_score_estimator(
            propensity_score,
            dataset;
            cache=cache,
            verbosity=verbosity,
            machine_cache=machine_cache
        )
        # Target Q̄ estimator
        last_cv_fold_candidate = last_cv_candidate[fold_index]
        targeted_η̂ₙ_train, _ = get_new_targeted_candidate(
            last_cv_fold_candidate, 
            fold_propensity_score_estimate,
            fluctuation_model, 
            dataset;
            use_fluct=use_fluct,
            verbosity=verbosity,
            cache=cache,
            machine_cache=machine_cache
        )
        validation_loss += loss_sum(targeted_η̂ₙ_train, selectrows(dataset, fold_train_val_indices[2]))
        push!(fold_candidates, targeted_η̂ₙ_train)
    end
    return fold_candidates, validation_loss
end

loss_sum(candidate, dataset) =
    sum(compute_loss(candidate.outcome_mean, dataset))

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

function compute_loss(conditional_density_estimate, dataset)
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