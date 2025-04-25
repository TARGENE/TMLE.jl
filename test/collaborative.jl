module TestCollaborative

using Test
using TMLE
using MLJLinearModels
using MLJBase

TEST_DIR = joinpath(pkgdir(TMLE), "test")
include(joinpath(TEST_DIR, "counterfactual_mean_based", "interactions_simulations.jl"))

@testset "Integration Test AdaptiveCorrelationOrdering" begin
    dataset, Ψ₀ = continuous_outcome_binary_treatment_pb(n=100_000)
    Ψ = AIE(
        outcome = :Y,
        treatment_values = (
            T₁=(case=true, control=false), 
            T₂=(case=true, control=false)
        ),
        treatment_confounders = (
            T₁=[:W₁, :W₂, :W₃],
            T₂=[:W₁, :W₂, :W₃],
        )
    )
    # Define the estimator
    cache = Dict()
    resampling = StratifiedCV()
    machine_cache = false
    verbosity = 1
    collaborative_strategy = AdaptiveCorrelationOrdering(resampling=StratifiedCV())
    train_validation_indices = MLJBase.train_test_pairs(resampling, 1:nrows(dataset), dataset, dataset.Y)
    tmle = TMLEE(;
        models = default_models(;
            Q_continuous = LinearRegressor(),
            G = LogisticClassifier(lambda=0.)
        ),
        collaborative_strategy = collaborative_strategy
    )
    # Initialize the relevant factors: no confounder is present in the propensity score
    η = TMLE.get_relevant_factors(Ψ, collaborative_strategy=collaborative_strategy)
    @test η.propensity_score == (TMLE.ConditionalDistribution(:T₁, (:COLLABORATIVE_INTERCEPT, :T₂)), TMLE.ConditionalDistribution(:T₂, (:COLLABORATIVE_INTERCEPT,)))
    # Estimate the initial factors: check models have been fitted correctly anc cahche is updated
    initial_factors_estimator = TMLE.CMRelevantFactorsEstimator(tmle.resampling, tmle.models)
    η̂ₙ = initial_factors_estimator(η, dataset; 
        cache=cache, 
        verbosity=verbosity, 
        machine_cache=tmle.machine_cache
    )
    for ps_component in η̂ₙ.propensity_score.components
        fp = fitted_params(ps_component.machine)
        fitted_variables = first.(fp.logistic_classifier.coefs)
        @test issubset(fitted_variables, [:COLLABORATIVE_INTERCEPT, :T₂__false])
    end
    @test haskey(cache, η.outcome_mean)
    for ps_component in η.propensity_score
        @test haskey(cache, ps_component)
    end
    # Collaborative Targeted Estimation
    estimator = TMLE.TargetedCMRelevantFactorsEstimator(
        Ψ, 
        η̂ₙ;
        collaborative_strategy=tmle.collaborative_strategy,
        tol=tmle.tol,
        max_iter=tmle.max_iter,
        ps_lowerbound=tmle.ps_lowerbound,
        weighted=tmle.weighted,
        machine_cache=tmle.machine_cache
    )
    fluctuation_model = estimator.fluctuation
    ## Check models are correctly retrieved
    models = TMLE.retrieve_models(estimator)
    @test models == Dict(
        :T₁ => tmle.models[:G_default],
        :T₂ => tmle.models[:G_default],
        :Y => tmle.models[:Q_continuous_default]
    )
    ## Check initialisation of the collaborative strategy
    TMLE.initialise!(estimator.collaborative_strategy, Ψ)
    @test estimator.collaborative_strategy.remaining_confounders == Set([:W₁, :W₃, :W₂])
    @test estimator.collaborative_strategy.current_confounders == Set()
    ## Check candidates initialisation: 
    ## - η corresponds to the propensity score with no confounders
    ## - The fluctuation is fitted on the whole dataset
    ## - A vector of (candidate, loss) is returned
    ## - A candidate is a targeted estimate
    candidates = TMLE.initialise_candidates(η, fluctuation_model, dataset;
        verbosity=verbosity,
        cache=cache,
        machine_cache=machine_cache
    )
    candidate, loss = only(candidates)
    @test candidate isa TMLE.MLCMRelevantFactors
    @test candidate.outcome_mean.machine.model isa TMLE.Fluctuation
    @test candidate.propensity_score === η̂ₙ.propensity_score
    @test loss == sum((abs.(TMLE.expected_value(candidate.outcome_mean, dataset) .- dataset.Y))) # The RMSE

    # Check CV candidates initialisation:
    ## - models are fitted on the training sets
    ## - The evaluation is made on the validation sets
    cv_candidates = TMLE.initialise_cv_candidates(η, dataset, fluctuation_model, train_validation_indices, models;
        cache=cache,
        verbosity=verbosity,
        machine_cache=true
    );
    cv_candidate = only(cv_candidates)
    validation_losses = []
    for (fold_estimate, (train_indices, val_indices)) in zip(cv_candidate.candidate, train_validation_indices)
        # Check Fluctuation
        targeted_outcome_mean_estimate = fold_estimate.outcome_mean
        _, y_train = targeted_outcome_mean_estimate.machine.data
        @test y_train == dataset.Y[train_indices]
        # CHeck Initial outcome mean estimate
        outcome_mean_estimate = fold_estimate.outcome_mean.machine.model.initial_factors.outcome_mean
        _, y_train = outcome_mean_estimate.machine.data
        @test y_train == dataset.Y[train_indices]
        # Check propensity score
        for ps_component in fold_estimate.propensity_score.components
            _, y_train = ps_component.machine.data
            @test y_train == dataset[ps_component.estimand.outcome][train_indices]
        end
        # validation loss
        append!(validation_losses, TMLE.compute_loss(targeted_outcome_mean_estimate, selectrows(dataset, val_indices)))
    end
    @test length(validation_losses) == nrows(dataset)
    @test sum(validation_losses) == cv_candidate.loss

    # We now enter the main loop
    # The collaborative strategy is updated
    TMLE.update!(collaborative_strategy, candidate, dataset)
    added_confounder = only(collaborative_strategy.current_confounders)
    @test length(collaborative_strategy.remaining_confounders) == 2
    # A new propensity score and associated estimator is suggested
    new_propensity_score, new_propensity_score_estimator = TMLE.get_new_propensity_score_and_estimator(
            collaborative_strategy, 
            Ψ, 
            dataset,
            models
    )
    @test all(added_confounder in ps_component.parents for ps_component in new_propensity_score)
    @test all(cde.train_validation_indices === nothing for cde in values(new_propensity_score_estimator.cd_estimators))
    new_propensity_score_estimate = new_propensity_score_estimator(
        new_propensity_score,
        dataset;
        cache=cache,
        verbosity=verbosity,
        machine_cache=machine_cache
    )
    # get new targeted candidate, not using the fluctuation model first
    new_candidate, new_loss = TMLE.get_new_targeted_candidate(candidate, new_propensity_score_estimate, fluctuation_model, dataset;
            use_fluct=false,
            verbosity=verbosity,
            cache=cache,
            machine_cache=machine_cache
    )
    outcome_mean_estimate_used = new_candidate.outcome_mean.machine.model.initial_factors.outcome_mean
    @test outcome_mean_estimate_used === candidate.outcome_mean.machine.model.initial_factors.outcome_mean
    fitted_propensity_score = new_candidate.outcome_mean.machine.model.initial_factors.propensity_score.estimand
    @test fitted_propensity_score === new_propensity_score

    # get new targeted candidate, this time using the fluctuation model
    new_candidate, new_loss = TMLE.get_new_targeted_candidate(candidate, new_propensity_score_estimate, fluctuation_model, dataset;
            use_fluct=true,
            verbosity=verbosity,
            cache=cache,
            machine_cache=machine_cache
    )
    outcome_mean_estimate_used = new_candidate.outcome_mean.machine.model.initial_factors.outcome_mean
    @test outcome_mean_estimate_used === candidate.outcome_mean
    ## The loss should be smaller because we fluctuate through the previous model
    @test new_loss < loss
    
    # Evaluate the new candidate in CV passing through the fluctuated model
    TMLE.evaluate_cv_candidate!(cv_candidates, fluctuation_model, new_propensity_score, models, dataset, train_validation_indices; 
        use_fluct=true,
        verbosity=verbosity,
        cache=cache,
        machine_cache=machine_cache
    )
    @test length(cv_candidates) == 2
    first_cv_dandidate = cv_candidates[1].candidate
    last_cv_candidate = last(cv_candidates).candidate
    for (fold_id, fold_candidate) in enumerate(last_cv_candidate)
        @test fold_candidate.outcome_mean.machine.model isa TMLE.Fluctuation
        @test fold_candidate.estimand.propensity_score === new_propensity_score
        @test fold_candidate.outcome_mean.machine.model.initial_factors.outcome_mean === first_cv_dandidate[fold_id].outcome_mean
    end
    @test TMLE.compute_validation_loss(last_cv_candidate, dataset, train_validation_indices) == cv_candidates[2].loss

    # Evaluate the new candidate in CV NOT passing through the fluctuated model
    TMLE.evaluate_cv_candidate!(cv_candidates, fluctuation_model, new_propensity_score, models, dataset, train_validation_indices; 
        use_fluct=false,
        verbosity=verbosity,
        cache=cache,
        machine_cache=machine_cache
    )
    @test length(cv_candidates) == 3
    second_cv_dandidate = cv_candidates[2].candidate
    last_cv_candidate = last(cv_candidates).candidate
    for (fold_id, fold_candidate) in enumerate(last_cv_candidate)
        @test fold_candidate.outcome_mean.machine.model isa TMLE.Fluctuation
        @test fold_candidate.estimand.propensity_score === new_propensity_score
        @test fold_candidate.outcome_mean.machine.model.initial_factors.outcome_mean === second_cv_dandidate[fold_id].outcome_mean.machine.model.initial_factors.outcome_mean
    end
    @test TMLE.compute_validation_loss(last_cv_candidate, dataset, train_validation_indices) == cv_candidates[2].loss

end

end

true