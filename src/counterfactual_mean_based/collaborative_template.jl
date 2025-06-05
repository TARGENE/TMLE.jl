#####################################################################
###                        CollaborativeStrategy                  ###
#####################################################################

"""
    CollaborativeStrategy

A collaborative strategy must implement the interface.
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
update!(strategy::CollaborativeStrategy, last_targeted_η̂ₙ, dataset) = error("Not Implemented Error.")

"""
CLeans the collaborative strategy.
"""
finalise!(strategy::CollaborativeStrategy) = error("Not Implemented Error.")

"""
Returns `true` when there is no more propensity score candidate to explore.
"""
exhausted(strategy::CollaborativeStrategy) = error("Not Implemented Error.")

"""
Each collaborative strategy must implement an `iterate` method iterating over (g, ĝ) candidates at step k of the algorithm. 
- For a general pattern see the `GreedyStrategy` implementation.
- For pre-ordered strategies, this iterator will stop after 1 iteration (see the `AdaptiveCorrelationStrategy`).
"""
struct StepKPropensityScoreIterator{T<:CollaborativeStrategy}
    collaborative_strategy::T
    Ψ
    dataset
    models
    last_targeted_η̂ₙ
end

Base.IteratorSize(iter::StepKPropensityScoreIterator) = Base.SizeUnknown()

#####################################################################
###                           Functions                           ###
#####################################################################

function step_k_best_candidate(
    collaborative_strategy,
    Ψ,
    dataset,
    models,
    fluctuation_model,
    last_targeted_η̂ₙ,
    last_loss;
    verbosity=1,
    cache=Dict(),
    machine_cache=false,
    acceleration=CPU1()
    )
    best = (g=nothing, ĝ=nothing, targeted_η̂ₙ=nothing, loss=Inf)
    use_fluct = false
    ps_iterator = StepKPropensityScoreIterator(collaborative_strategy, Ψ, dataset, models, last_targeted_η̂ₙ)
    ps_sequence = map(ps_iterator) do (g, ĝ)
        # Fit the new propensity score
        ĝₙ = ĝ(
            g,
            dataset;
            cache=cache,
            verbosity=verbosity-1,
            machine_cache=machine_cache,
            acceleration=acceleration
        )
        # Fluctuate outcome model through the new propensity score and Q̄k
        targeted_η̂ₙ, loss = get_new_targeted_candidate(last_targeted_η̂ₙ, ĝₙ, fluctuation_model, dataset;
            use_fluct=use_fluct,
            verbosity=verbosity-1,
            cache=cache,
            machine_cache=machine_cache
        )
        # Update best
        if loss < best.loss
            best = (g=g, ĝ=ĝ, targeted_η̂ₙ=targeted_η̂ₙ, loss=loss)
        end
        # Store candidate
        (g, ĝ, ĝₙ)
    end

    # If the loss is not decreased, we repeat by fluctuating through Q̄k,* from the previous candidate's flutuated model
    if best.loss > last_loss
        best = (g=nothing, ĝ=nothing, targeted_η̂ₙ=nothing, loss=Inf)
        use_fluct = true
        for (g, ĝ, ĝₙ) in ps_sequence
            targeted_η̂ₙ, loss = get_new_targeted_candidate(last_targeted_η̂ₙ, ĝₙ, fluctuation_model, dataset;
                use_fluct=use_fluct,
                verbosity=verbosity-1,
                cache=cache,
                machine_cache=machine_cache
            )
            # Update best
            if loss < best.loss
                best = (g=g, ĝ=ĝ, targeted_η̂ₙ=targeted_η̂ₙ, loss=loss)
            end
        end
    end
    # Return the best candidate and whether the seqeunce goes trough the fluctuation model
    return best.g, best.ĝ, best.targeted_η̂ₙ, best.loss, use_fluct
end

function find_optimal_candidate(
    last_candidate_info, 
    collaborative_strategy, 
    Ψ, 
    dataset, 
    fluctuation_model, 
    train_validation_indices, 
    models;
    verbosity=1,
    cache=Dict(),
    machine_cache=false,
    acceleration=CPU1()
    )
    candidate_id = 1
    best_candidate = (targeted_η̂ₙ=last_candidate_info.targeted_η̂ₙ, cv_loss=last_candidate_info.cv_loss, id=candidate_id)
    verbosity > 0 && @info @sprintf("Initial candidate's CV loss: %.5f", best_candidate.cv_loss)
    while !exhausted(collaborative_strategy)
        candidate_id += 1
        last_targeted_η̂ₙ, last_loss, last_cv_targeted_η̂ₙ, last_cv_loss = last_candidate_info
        # Find best candidate
        new_g, new_ĝ, new_targeted_η̂ₙ, new_loss, use_fluct = step_k_best_candidate(
            collaborative_strategy,
            Ψ,
            dataset,
            models,
            fluctuation_model,
            last_targeted_η̂ₙ,
            last_loss;
            verbosity=verbosity,
            cache=cache,
            machine_cache=machine_cache,
            acceleration=acceleration
        )
        # Update the collaborative strategy's state
        update!(collaborative_strategy, new_g, new_ĝ)
        # Evaluate candidate
        new_cv_targeted_η̂ₙ, new_cv_loss = evaluate_cv_candidate(last_cv_targeted_η̂ₙ, fluctuation_model, new_g, models, dataset, train_validation_indices; 
            use_fluct=use_fluct,
            verbosity=verbosity-1,
            cache=cache,
            machine_cache=machine_cache,
            acceleration=acceleration
        )
        # Update last candidate info
        last_candidate_info = (targeted_η̂ₙ=new_targeted_η̂ₙ, loss=new_loss, cv_candidate=new_cv_targeted_η̂ₙ, cv_loss=new_cv_loss)

        # Update the best candidate or early stop
        if new_cv_loss < best_candidate.cv_loss
            verbosity > 0 && @info @sprintf("New candidate's CV loss: %.5f, updating best candidate.", new_cv_loss)
            best_candidate = (targeted_η̂ₙ=new_targeted_η̂ₙ, cv_loss=new_cv_loss, id=candidate_id)
        elseif candidate_id - best_candidate.id > collaborative_strategy.patience
            verbosity > 0 && @info @sprintf("New candidate's CV loss: %.5f, patience reached, terminating.",new_cv_loss)
            break
        else
            verbosity > 0 && @info @sprintf("New candidate's CV loss: %.5f, continuing.", new_cv_loss)
        end
    end
    return best_candidate
end

function get_initial_candidate(η, fluctuation_model, dataset;
    verbosity=1,
    cache=Dict(),
    machine_cache=false
    )
    targeted_η̂ = CMBasedTMLE(
        fluctuation_model, 
        nothing,
    )
    targeted_η̂ₙ = targeted_η̂(η, dataset;
        cache=cache,
        verbosity=verbosity,
        machine_cache=machine_cache
    )
    loss = mean_loss(targeted_η̂ₙ, dataset)
    return (targeted_η̂ₙ=targeted_η̂ₙ, loss=loss)
end

function get_new_targeted_candidate(last_targeted_η̂ₙ, new_propensity_score_estimate, fluctuation_model, dataset;
    use_fluct=false,
    verbosity=1,
    cache=Dict(),
    machine_cache=false
    )
    new_η = TMLE.CMRelevantFactors(last_targeted_η̂ₙ.estimand.outcome_mean, new_propensity_score_estimate.estimand)
    # Define new nuisance factors estimate
    η̂ₙ = TMLE.MLCMRelevantFactors(
        new_η, 
        use_fluct ? last_targeted_η̂ₙ.outcome_mean : last_targeted_η̂ₙ.outcome_mean.machine.model.initial_factors.outcome_mean, 
        new_propensity_score_estimate
    )
    # Fluctuate
    new_fluctuation = Fluctuation(fluctuation_model.Ψ, η̂ₙ; 
        tol=fluctuation_model.tol,
        max_iter=fluctuation_model.max_iter, 
        ps_lowerbound=fluctuation_model.ps_lowerbound, 
        weighted=fluctuation_model.weighted,
        cache=fluctuation_model.cache
    )
    targeted_η̂ = TMLE.CMBasedTMLE(new_fluctuation, nothing)
    targeted_η̂ₙ = targeted_η̂(new_η, dataset;
        cache=cache,
        verbosity=verbosity,
        machine_cache=machine_cache
    )
    loss = mean_loss(targeted_η̂ₙ, dataset)
    return targeted_η̂ₙ, loss
end

function get_initial_cv_candidate(η, dataset, fluctuation_model, train_validation_indices, models;
    cache=Dict(),
    verbosity=1,
    machine_cache=false,
    acceleration=CPU1()
    )
    # Estimate Nuisance parameters on each fold
    η̂ = TMLE.FoldsCMRelevantFactorsEstimator(models, train_validation_indices)
    η̂ₙ = η̂(η, dataset;
        cache=cache,
        verbosity=verbosity,
        machine_cache=machine_cache,
        acceleration=acceleration
    )
    # Target Nuisance parameters on each fold
    targeted_η̂ = TMLE.CMBasedFoldsTMLE(
            fluctuation_model.Ψ, 
            η̂ₙ,
            train_validation_indices;
            tol=fluctuation_model.tol,
            max_iter=fluctuation_model.max_iter,
            ps_lowerbound=fluctuation_model.ps_lowerbound,
            weighted=fluctuation_model.weighted,
            machine_cache=machine_cache
        )
    targeted_η̂ₙ = targeted_η̂(η, dataset;
            cache=cache,
            verbosity=verbosity,
            machine_cache=machine_cache,
            acceleration=acceleration
        )
    # Evaluate candidate on validation fold
    validation_loss = compute_validation_loss(targeted_η̂ₙ, dataset, train_validation_indices)

    return (targeted_η̂ₙ=targeted_η̂ₙ, loss=validation_loss)
end

function compute_validation_loss(targeted_η̂ₙ, dataset, train_validation_indices)
    folds_val_losses = map(zip(train_validation_indices, targeted_η̂ₙ)) do ((_, val_indices), fold_targeted_η̂ₙ)
        validation_dataset = selectrows(dataset, val_indices)
        mean_loss(fold_targeted_η̂ₙ, validation_dataset)
    end
    return mean(folds_val_losses)
end

function update_cv_folds_info!(
    folds_targeted_η̂ₙ, 
    folds_val_losses,
    last_cv_targeted_η̂ₙ,
    train_validation_indices, 
    fluctuation_model,
    propensity_score,
    models,
    dataset,
    fold_index;
    use_fluct=false,
    verbosity=1,
    cache=Dict(),
    machine_cache=false
    )
    fold_train_val_indices = train_validation_indices[fold_index]
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
    last_cv_fold_targeted_η̂ₙ = last_cv_targeted_η̂ₙ[fold_index]
    targeted_η̂ₙ_train, _ = get_new_targeted_candidate(
        last_cv_fold_targeted_η̂ₙ, 
        fold_propensity_score_estimate,
        fluctuation_model, 
        dataset;
        use_fluct=use_fluct,
        verbosity=verbosity,
        cache=cache,
        machine_cache=machine_cache
    )
    folds_val_losses[fold_index] = mean_loss(targeted_η̂ₙ_train, selectrows(dataset, fold_train_val_indices[2]))
    folds_targeted_η̂ₙ[fold_index] = targeted_η̂ₙ_train
end

function fill_cv_folds_info!(acceleration::CPU1, folds_targeted_η̂ₙ, args...; kwargs...)
    nfolds = length(folds_targeted_η̂ₙ)
    for fold_index in 1:nfolds
        update_cv_folds_info!(folds_targeted_η̂ₙ, args..., fold_index;kwargs...)
    end
end

function fill_cv_folds_info!(acceleration::CPUThreads, folds_targeted_η̂ₙ, args...; kwargs...)
    nfolds = length(folds_targeted_η̂ₙ)
    @threads for fold_index in 1:nfolds
        update_cv_folds_info!(folds_targeted_η̂ₙ, args..., fold_index;kwargs...)
    end
end

function evaluate_cv_candidate(last_cv_targeted_η̂ₙ, fluctuation_model, propensity_score, models, dataset, train_validation_indices; 
    use_fluct=false,
    verbosity=1,
    cache=Dict(),
    machine_cache=false,
    acceleration=CPU1()
    )
    n_folds = length(train_validation_indices)
    folds_targeted_η̂ₙ = Vector{Any}(undef, n_folds)
    folds_val_losses = Vector{Float64}(undef, n_folds)
    fill_cv_folds_info!(acceleration,
        folds_targeted_η̂ₙ, 
        folds_val_losses,
        last_cv_targeted_η̂ₙ,
        train_validation_indices, 
        fluctuation_model,
        propensity_score,
        models,
        dataset;
        use_fluct=use_fluct,
        verbosity=verbosity,
        cache=cache,
        machine_cache=machine_cache
    )
    return folds_targeted_η̂ₙ, mean(folds_val_losses)
end

mean_loss(η̂ₙ, dataset) =
    mean(compute_loss(η̂ₙ.outcome_mean, dataset))

adapt_and_getloss(ŷ::Vector{<:Real}) = ŷ, RootMeanSquaredError()

adapt_and_getloss(ŷ::Vector{<:Distribution{Univariate, Distributions.Continuous}}) = TMLE.expected_value(ŷ), RootMeanSquaredError()

adapt_and_getloss(ŷ::UnivariateFiniteVector) = ŷ, LogLoss()

function compute_loss(conditional_density_estimate, dataset)
    ŷ = MLJBase.predict(conditional_density_estimate, dataset)
    y = dataset[!, conditional_density_estimate.estimand.outcome]
    ŷ, loss = TMLE.adapt_and_getloss(ŷ)
    return measurements(loss, ŷ, y)
end
