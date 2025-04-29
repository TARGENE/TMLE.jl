module TestCollaborative

using Test
using TMLE
using MLJLinearModels
using MLJBase

TEST_DIR = joinpath(pkgdir(TMLE), "test")
include(joinpath(TEST_DIR, "counterfactual_mean_based", "interactions_simulations.jl"))

@testset "Test Interface" begin
    dataset, Ψ₀ = continuous_outcome_binary_treatment_pb(n=1_000)
    Ψ = AIE(
        outcome = :Y,
        treatment_values = (
            T₁=(case=true, control=false), 
            T₂=(case=true, control=false)
        ),
        treatment_confounders = (
            T₁=[:W₁, :W₂],
            T₂=[:W₁, :W₂],
        )
    )
    # Define the strategy
    adaptive_strategy = AdaptiveCorrelationOrdering()
    @test adaptive_strategy.patience == 10
    @test adaptive_strategy.remaining_confounders == Set{Symbol}()
    @test adaptive_strategy.current_confounders == Set{Symbol}()
    ps = TMLE.propensity_score(Ψ, adaptive_strategy)
    @test ps == (
        TMLE.ConditionalDistribution(:T₁, (:COLLABORATIVE_INTERCEPT, :T₂)), 
        TMLE.ConditionalDistribution(:T₂, (:COLLABORATIVE_INTERCEPT,))
    )
    # Initialisation
    TMLE.initialise!(adaptive_strategy, Ψ)
    @test adaptive_strategy.remaining_confounders == Set([:W₁, :W₂])
    @test adaptive_strategy.current_confounders == Set()
    # Updates
    estimator = TMLE.MLConditionalDistributionEstimator(LinearRegressor(), nothing)
    cde = estimator(
        TMLE.ConditionalDistribution(:Y, (:COLLABORATIVE_INTERCEPT,)),
        dataset,
    )
    last_candidate = (outcome_mean=cde,)
    loss = TMLE.compute_loss(cde, dataset)
    @test TMLE.exhausted(adaptive_strategy) == false
    @test abs(cor(dataset.W₁, loss)) > abs(cor(dataset.W₂, loss))
    TMLE.update!(adaptive_strategy, last_candidate, dataset)
    @test adaptive_strategy.remaining_confounders == Set([:W₂])
    @test adaptive_strategy.current_confounders == Set([:W₁])
    @test TMLE.exhausted(adaptive_strategy) == false
    ps = TMLE.propensity_score(Ψ, adaptive_strategy)
    @test ps == (
        TMLE.ConditionalDistribution(:T₁, (:COLLABORATIVE_INTERCEPT, :T₂, :W₁)), 
        TMLE.ConditionalDistribution(:T₂, (:COLLABORATIVE_INTERCEPT, :W₁))
    )
    TMLE.update!(adaptive_strategy, last_candidate, dataset)
    @test adaptive_strategy.remaining_confounders == Set()
    @test adaptive_strategy.current_confounders == Set([:W₁, :W₂])
    ps = TMLE.propensity_score(Ψ, adaptive_strategy)
    @test ps == (
        TMLE.ConditionalDistribution(:T₁, (:COLLABORATIVE_INTERCEPT, :T₂, :W₁, :W₂)), 
        TMLE.ConditionalDistribution(:T₂, (:COLLABORATIVE_INTERCEPT, :W₁, :W₂))
    )
    @test TMLE.exhausted(adaptive_strategy) == true
    # Finalisation
    TMLE.finalise!(adaptive_strategy)
    @test adaptive_strategy.remaining_confounders == Set{Symbol}()
    @test adaptive_strategy.current_confounders == Set{Symbol}()
end

@testset "Integration Test AdaptiveCorrelationOrdering" begin
    n_samples = 1_000
    dataset, Ψ₀ = continuous_outcome_binary_treatment_pb(n=1_000)
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
    machine_cache = true
    verbosity = 0
    collaborative_strategy = AdaptiveCorrelationOrdering()
    tmle = Tmle(;
        models = default_models(;
            Q_continuous = LinearRegressor(),
            G = LogisticClassifier(lambda=0.)
        ),
        resampling=resampling,
        collaborative_strategy = collaborative_strategy,
        machine_cache=machine_cache,
    )
    # Initialize the relevant factors: no confounder is present in the propensity score
    η = TMLE.get_relevant_factors(Ψ, collaborative_strategy=collaborative_strategy)
    @test η.propensity_score == (
        TMLE.ConditionalDistribution(:T₁, (:COLLABORATIVE_INTERCEPT, :T₂)), 
        TMLE.ConditionalDistribution(:T₂, (:COLLABORATIVE_INTERCEPT,))
    )
    # Estimate the initial factors: check models have been fitted correctly and cahche is updated
    train_validation_indices = MLJBase.train_test_pairs(resampling, 1:nrows(dataset), dataset, dataset.Y)
    initial_factors_estimator = TMLE.CMRelevantFactorsEstimator(tmle.collaborative_strategy; 
        train_validation_indices=train_validation_indices, 
        models=tmle.models
    )
    η̂ₙ = initial_factors_estimator(η, dataset; 
        cache=cache, 
        verbosity=verbosity, 
        machine_cache=tmle.machine_cache
    )
    for ps_component in η̂ₙ.propensity_score.components
        fp = fitted_params(ps_component.machine)
        X, _ = ps_component.machine.data
        @test nrows(X) == n_samples
        fitted_variables = first.(fp.logistic_classifier.coefs)
        @test issubset(fitted_variables, [:COLLABORATIVE_INTERCEPT, :T₂__false])
    end
    ## One estimate for each estimand in the cache
    @test length(cache) == 4
    for estimand in (η, η.outcome_mean, η.propensity_score...)
        @test length(cache[estimand]) == 1
    end
    # Collaborative Targeted Estimation
    estimator = TMLE.TargetedCMRelevantFactorsEstimator(
        Ψ, 
        η̂ₙ;
        collaborative_strategy=tmle.collaborative_strategy,
        train_validation_indices=train_validation_indices,
        tol=tmle.tol,
        max_iter=tmle.max_iter,
        ps_lowerbound=tmle.ps_lowerbound,
        weighted=tmle.weighted,
        machine_cache=tmle.machine_cache
    )
    @test estimator isa TMLE.TargetedCMRelevantFactorsEstimator{AdaptiveCorrelationOrdering}
    @test estimator.train_validation_indices == train_validation_indices
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
    @test nrows(candidate.outcome_mean.machine.data[1]) == n_samples
    @test candidate.propensity_score === η̂ₙ.propensity_score
    @test loss == TMLE.loss_sum(candidate, dataset)

    # Check CV candidates initialisation:
    ## - models are fitted on the training sets
    ## - The evaluation is made on the validation sets
    ## - The cache will contain all estimates to be reused
    cv_candidates = TMLE.initialise_cv_candidates(η, dataset, fluctuation_model, train_validation_indices, models;
        cache=cache,
        verbosity=2,
        machine_cache=machine_cache
    );
    cv_candidate, cv_loss = only(cv_candidates)
    validation_losses = []
    for (fold_estimate, (train_indices, val_indices)) in zip(cv_candidate, train_validation_indices)
        # Check the estimate is in the cache
        @test fold_estimate.outcome_mean.machine.model.initial_factors in values(cache[fold_estimate.estimand])
        # Check Fluctuation
        targeted_outcome_mean_estimate = fold_estimate.outcome_mean
        _, y_train = targeted_outcome_mean_estimate.machine.data
        @test y_train == dataset.Y[train_indices]
        # Check Initial outcome mean estimate
        outcome_mean_estimate = fold_estimate.outcome_mean.machine.model.initial_factors.outcome_mean
        @test outcome_mean_estimate in values(cache[outcome_mean_estimate.estimand])
        _, y_train = outcome_mean_estimate.machine.data
        @test y_train == dataset.Y[train_indices]
        # Check propensity score
        for ps_component in fold_estimate.propensity_score.components
            _, y_train = ps_component.machine.data
            @test y_train == dataset[ps_component.estimand.outcome][train_indices]
            @test ps_component in values(cache[ps_component.estimand])
        end
        # validation loss
        append!(validation_losses, TMLE.compute_loss(targeted_outcome_mean_estimate, selectrows(dataset, val_indices)))
    end
    @test length(validation_losses) == n_samples
    @test sum(validation_losses) ≈ cv_loss

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
    for ps_component in new_propensity_score_estimate.components
        @test nrows(ps_component.machine.data[1]) == n_samples
        @test ps_component in values(cache[ps_component.estimand])
    end
    # get new targeted candidate, not using the fluctuation model first
    new_candidate, new_loss = TMLE.get_new_targeted_candidate(candidate, new_propensity_score_estimate, fluctuation_model, dataset;
            use_fluct=false,
            verbosity=verbosity,
            cache=cache,
            machine_cache=machine_cache
    )
    # The outcome_mean model which is fluctuated through is Q̄n,k
    outcome_mean_estimate_used = new_candidate.outcome_mean.machine.model.initial_factors.outcome_mean
    @test outcome_mean_estimate_used === candidate.outcome_mean.machine.model.initial_factors.outcome_mean
    # The propensity score used for fluctuation is the new candidate
    fitted_propensity_score = new_candidate.outcome_mean.machine.model.initial_factors.propensity_score.estimand
    @test fitted_propensity_score === new_propensity_score

    # get new targeted candidate, this time using the fluctuation model
    new_candidate, new_loss_bis = TMLE.get_new_targeted_candidate(candidate, new_propensity_score_estimate, fluctuation_model, dataset;
            use_fluct=true,
            verbosity=verbosity,
            cache=cache,
            machine_cache=machine_cache
    )
    # The outcome_mean model which is fluctuated through is Q̄n,k,*
    outcome_mean_estimate_used = new_candidate.outcome_mean.machine.model.initial_factors.outcome_mean
    @test outcome_mean_estimate_used === candidate.outcome_mean
    ## The loss should be smaller because we fluctuate through the previous model, however in finite samples
    ## I suppose this is not warranted, we check they are approximately equal and the loss with  Q̄n,k,* <  Q̄n,k
    @test loss ≈ new_loss_bis atol=1e-2
    @test new_loss_bis < new_loss
    
    # Evaluate the new candidate in CV passing through the fluctuated model
    second_cv_candidate, second_cvloss = TMLE.evaluate_cv_candidate!(cv_candidate, fluctuation_model, new_propensity_score, models, dataset, train_validation_indices; 
        use_fluct=true,
        verbosity=verbosity,
        cache=cache,
        machine_cache=machine_cache
    )
    for (fold_id, fold_candidate) in enumerate(second_cv_candidate)
        @test fold_candidate.outcome_mean.machine.model isa TMLE.Fluctuation
        @test fold_candidate.estimand.propensity_score === new_propensity_score
        @test fold_candidate.outcome_mean.machine.model.initial_factors.outcome_mean === cv_candidate[fold_id].outcome_mean
    end
    @test TMLE.compute_validation_loss(second_cv_candidate, dataset, train_validation_indices) == second_cvloss

    # Evaluate the new candidate in CV NOT passing through the fluctuated model
    second_cv_candidate_bis, second_cvloss_bis = TMLE.evaluate_cv_candidate!(cv_candidate, fluctuation_model, new_propensity_score, models, dataset, train_validation_indices; 
        use_fluct=false,
        verbosity=verbosity,
        cache=cache,
        machine_cache=machine_cache
    )
    for (fold_id, fold_candidate) in enumerate(second_cv_candidate_bis)
        @test fold_candidate.outcome_mean.machine.model isa TMLE.Fluctuation
        @test fold_candidate.estimand.propensity_score === new_propensity_score
        @test fold_candidate.outcome_mean.machine.model.initial_factors.outcome_mean === cv_candidate[fold_id].outcome_mean.machine.model.initial_factors.outcome_mean
    end
    @test TMLE.compute_validation_loss(second_cv_candidate_bis, dataset, train_validation_indices) == second_cvloss_bis

    # Check the full loop
    best_candidate = TMLE.update_candidates!(
        candidates, 
        cv_candidates, 
        collaborative_strategy, 
        Ψ, 
        dataset, 
        fluctuation_model, 
        train_validation_indices, 
        models;
        verbosity=0,
        cache=Dict(),
        machine_cache=false
    )
    best_loss, best_index = findmin(x -> x.loss, cv_candidates)
    @test best_candidate.id == best_index
    @test best_candidate.loss == best_loss
    @test best_candidate.candidate == candidates[best_index].candidate
end

end

true