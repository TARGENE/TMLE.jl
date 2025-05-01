#####################################################################
###              Targeting AdaptiveCorrelationOrdering            ###
#####################################################################

"""
    CollaborativeStrategy

A collaborative strategy must implement the interface
"""
abstract type CollaborativeStrategy end

"""
Create a propensity score `JointConditionalDistribution` given the current state of `collaborative_strategy`
"""
propensity_score(Ψ, collaborative_strategy::CollaborativeStrategy) = error("Not Implemented Error.")

"""
Initialises the collaborative strategy
"""
initialise!(strategy::CollaborativeStrategy, Ψ) = error("Not Implemented Error.")

"""
Updates the collaborative strategy with the last candidate
"""
update!(strategy::CollaborativeStrategy, last_candidate, dataset) = error("Not Implemented Error.")

"""
CLeans the collaborative strategy.
"""
finalise!(strategy::CollaborativeStrategy) = error("Not Implemented Error.")

"""
Returns `true` when there is no more propensity score candidate to explore.
"""
exhausted(strategy::CollaborativeStrategy) = error("Not Implemented Error.")

#####################################################################
###                           Functions                           ###
#####################################################################

function update_candidates!(
    candidates, 
    cv_candidates, 
    collaborative_strategy, 
    Ψ, 
    dataset, 
    fluctuation_model, 
    train_validation_indices, 
    models;
    verbosity=1,
    cache=Dict(),
    machine_cache=false
    )
    candidate_id = 1
    best_candidate = (candidate=only(candidates).candidate, cvloss=only(cv_candidates).loss, id=candidate_id)
    verbosity > 0 && @info "Initial candidate's CV loss: $(best_candidate.cvloss)"
    while !exhausted(collaborative_strategy)
        candidate_id += 1
        # Update the collaborative strategy's state
        last_candidate, last_loss = last(candidates)
        update!(collaborative_strategy, last_candidate, dataset)
        # Get the new propensity score and associated estimator
        new_propensity_score, new_propensity_score_estimator = get_new_propensity_score_and_estimator(
            collaborative_strategy, 
            Ψ, 
            dataset,
            models
        )
        # Fit this new propensity score
        new_propensity_score_estimate = new_propensity_score_estimator(
            new_propensity_score,
            dataset;
            cache=cache,
            verbosity=verbosity-1,
            machine_cache=machine_cache
        )
        # Fluctuate outcome model through the new propensity score and Q̄k
        use_fluct = false
        candidate, loss = get_new_targeted_candidate(last_candidate, new_propensity_score_estimate, fluctuation_model, dataset;
            use_fluct=use_fluct,
            verbosity=verbosity-1,
            cache=cache,
            machine_cache=machine_cache
        )
        if loss > last_loss
            use_fluct = true
            # Fluctuate through Q̄k,* from the previous candidate's flutuated model
            candidate, loss = get_new_targeted_candidate(last_candidate, new_propensity_score_estimate, fluctuation_model, dataset;
                use_fluct=use_fluct,
                verbosity=verbosity-1,
                cache=cache,
                machine_cache=machine_cache
            )
        end
        push!(candidates, (candidate=candidate, loss=loss))
        # Evaluate candidate
        last_cv_candidate, last_cv_loss = last(cv_candidates)
        cv_candidate, cv_loss = evaluate_cv_candidate!(last_cv_candidate, fluctuation_model, new_propensity_score, models, dataset, train_validation_indices; 
            use_fluct=use_fluct,
            verbosity=verbosity-1,
            cache=cache,
            machine_cache=machine_cache
        )
        push!(cv_candidates, (candidate=cv_candidate, loss=cv_loss))
        # Update the best candidate or early stop
        if cv_loss < best_candidate.cvloss
            verbosity > 0 && @info "New candidate's CV loss: $(cv_loss), updating best candidate."
            best_candidate = (candidate=candidate, cvloss=cv_loss, id=candidate_id)
        elseif candidate_id - best_candidate.id > collaborative_strategy.patience
            verbosity > 0 && @info "New candidate's CV loss: $(cv_loss), patience reached, terminating."
            break
        else
            verbosity > 0 && @info "New candidate's CV loss: $(cv_loss)."
        end
    end
    return best_candidate
end

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
    targeted_η̂ = TargetedCMRelevantFactorsEstimator(
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

adapt_and_getloss(ŷ::Vector{<:Real}) = ŷ, RootMeanSquaredError()

adapt_and_getloss(ŷ::Vector{<:Distribution{Univariate, Distributions.Continuous}}) = TMLE.expected_value(ŷ), RootMeanSquaredError()

adapt_and_getloss(ŷ::UnivariateFiniteVector) = ŷ, LogLoss()

function compute_loss(conditional_density_estimate, dataset)
    ŷ = MLJBase.predict(conditional_density_estimate, dataset)
    y = dataset[!, conditional_density_estimate.estimand.outcome]
    ŷ, loss = TMLE.adapt_and_getloss(ŷ)
    return measurements(loss, ŷ, y)
end
