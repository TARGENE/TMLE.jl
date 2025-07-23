module TestCollaborative

using Test
using TMLE
using MLJLinearModels
using MLJBase

TEST_DIR = joinpath(pkgdir(TMLE), "test")
include(joinpath(TEST_DIR, "counterfactual_mean_based", "interactions_simulations.jl"))

@testset "Test AdaptiveCorrelationStrategy Interface" begin
    dataset, Ψ₀ = continuous_outcome_binary_treatment_pb(n=1_000)
    Ψ = AIE(
        outcome = :Y,
        treatment_values = (
            T₁=(case=true, control=false), 
            T₂=(case=true, control=false)
        ),
        treatment_confounders = (
            T₁=[:W₁, :W₂],
            T₂=[:W₁, :W₃],
        )
    )
    # Define the strategy
    models = default_models()
    adaptive_strategy = AdaptiveCorrelationStrategy()
    @test adaptive_strategy.patience == 10
    @test adaptive_strategy.remaining_confounders == Set{Symbol}()
    @test adaptive_strategy.current_confounders == Set{Symbol}()
    g_init = TMLE.get_treatments_factor(Ψ, adaptive_strategy, models)
    @test g_init == TMLE.JointConditionalDistribution(
        TMLE.ConditionalDistribution(:T₁, (:T₂,)), 
        TMLE.ConditionalDistribution(:T₂, ())
    )
    # Initialisation
    TMLE.initialise!(adaptive_strategy, Ψ)
    @test adaptive_strategy.remaining_confounders == Set([:W₁, :W₂, :W₃])
    @test adaptive_strategy.current_confounders == Set()
    # Updates
    g_1 = TMLE.JointConditionalDistribution(
        TMLE.ConditionalDistribution(:T₁, (:T₂, :W₂)), 
        TMLE.ConditionalDistribution(:T₂, ())
    )
    TMLE.update!(adaptive_strategy, g_1, nothing)
    @test adaptive_strategy.remaining_confounders == Set([:W₁, :W₃])
    @test adaptive_strategy.current_confounders == Set([:W₂])
    @test TMLE.exhausted(adaptive_strategy) == false
    g_2 = TMLE.JointConditionalDistribution(
        TMLE.ConditionalDistribution(:T₁, (:T₂, :W₁, :W₂)), 
        TMLE.ConditionalDistribution(:T₂, (:W₁,))
    )
    TMLE.update!(adaptive_strategy, g_2, nothing)
    @test adaptive_strategy.remaining_confounders == Set([:W₃])
    @test adaptive_strategy.current_confounders == Set([:W₁, :W₂])
    @test TMLE.exhausted(adaptive_strategy) == false
    g_3 = TMLE.JointConditionalDistribution(
        TMLE.ConditionalDistribution(:T₁, (:T₂, :W₁, :W₂)), 
        TMLE.ConditionalDistribution(:T₂, (:W₁, :W₃))
    )
    TMLE.update!(adaptive_strategy, g_3, nothing)
    @test adaptive_strategy.remaining_confounders == Set{Symbol}()
    @test adaptive_strategy.current_confounders == Set([:W₁, :W₂, :W₃])
    @test TMLE.exhausted(adaptive_strategy) == true
    # Finalisation does nothing
    TMLE.finalise!(adaptive_strategy)
    @test adaptive_strategy.remaining_confounders == Set{Symbol}()
    @test adaptive_strategy.current_confounders == Set([:W₁, :W₂, :W₃])

    # Full run: this leads to only W₃ being used for the propensity score
    ctmle = Tmle(collaborative_strategy=adaptive_strategy)
    Ψ = AIE(
        outcome = :Y,
        treatment_values = (
            T₁=(case=true, control=false), 
            T₂=(case=true, control=false)
        ),
        treatment_confounders = [:W₁, :W₂, :W₃]
    )
    result_ctmle, cache = ctmle(Ψ, dataset;verbosity=0);
    targeted_η̂ = cache[:targeted_factors]
    @test targeted_η̂.treatments_factor[1].estimand == TMLE.ConditionalDistribution(:T₁, (:T₂, :W₃))
    @test targeted_η̂.treatments_factor[2].estimand == TMLE.ConditionalDistribution(:T₂, (:W₃,))
end

@testset "Test GreedyStrategy Interface" begin
    dataset, Ψ₀ = continuous_outcome_binary_treatment_pb(n=1_000)
    Ψ = AIE(
        outcome = :Y,
        treatment_values = (
            T₁=(case=true, control=false), 
            T₂=(case=true, control=false)
        ),
        treatment_confounders = (
            T₁=[:W₁, :W₂],
            T₂=[:W₁, :W₃],
        )
    )
    # Define the estimator
    cache = Dict()
    resampling = StratifiedCV()
    machine_cache = true
    verbosity = 0
    collaborative_strategy = GreedyStrategy()
    models = default_models(;
        Q_continuous = LinearRegressor(),
        G = LogisticClassifier(lambda=0.)
    )
    tmle = Tmle(;
        weighted=false,
        models=models,
        resampling=resampling,
        collaborative_strategy = collaborative_strategy,
        machine_cache=machine_cache,
    )
    # Initialisation
    TMLE.initialise!(collaborative_strategy, Ψ)
    @test collaborative_strategy.remaining_confounders == Set([:W₁, :W₂, :W₃])
    @test collaborative_strategy.current_confounders == Set{Symbol}()
    # Fit nuisance functions
    train_validation_indices = MLJBase.train_test_pairs(resampling, 1:nrows(dataset), dataset, dataset.Y)
    initial_factors_estimator = TMLE.CMRelevantFactorsEstimator(tmle.collaborative_strategy; 
        train_validation_indices=train_validation_indices, 
        models=tmle.models
    )
    η = TMLE.get_relevant_factors(Ψ, models, collaborative_strategy=collaborative_strategy)
    η̂ₙ = initial_factors_estimator(η, dataset;
        cache=cache, 
        verbosity=verbosity, 
        machine_cache=tmle.machine_cache
    )
    fluctuation_model = TMLE.Fluctuation(Ψ, η̂ₙ)
    targeted_η̂ₙ, loss = TMLE.get_initial_candidate(η, fluctuation_model, dataset;
        verbosity=verbosity,
        cache=cache,
        machine_cache=machine_cache
    )
    # Check the iterator iterates over all confounders
    step_k_candidate_iterator = TMLE.StepKPropensityScoreIterator(collaborative_strategy, Ψ, dataset, models, targeted_η̂ₙ)
    g_ĝ_candidates = collect(step_k_candidate_iterator)
    g_candidates = Set(first.(g_ĝ_candidates))
    @test g_candidates == Set([
        TMLE.JointConditionalDistribution(
            TMLE.ConditionalDistribution(:T₁, (:T₂, :W₁)), 
            TMLE.ConditionalDistribution(:T₂, (:W₁,))
        ),
        TMLE.JointConditionalDistribution(
            TMLE.ConditionalDistribution(:T₁, (:T₂,)), 
            TMLE.ConditionalDistribution(:T₂, (:W₃,))
        ),
        TMLE.JointConditionalDistribution(
            TMLE.ConditionalDistribution(:T₁, (:T₂, :W₂)),
            TMLE.ConditionalDistribution(:T₂, ())
        )
    ])
    ĝ = only(unique(last.(g_ĝ_candidates)))
    @test ĝ.cd_estimators[:T₁].model == models[:G_default]
    @test ĝ.cd_estimators[:T₂].model == models[:G_default]
    # Find optimal candidate at the first iteration
    new_g, new_ĝ, new_targeted_η̂ₙ, new_loss, use_fluct = TMLE.step_k_best_candidate(
            collaborative_strategy,
            Ψ,
            dataset,
            models,
            fluctuation_model,
            targeted_η̂ₙ,
            loss;
            verbosity=verbosity,
            cache=cache,
            machine_cache=machine_cache,
    )
    @test new_g == TMLE.JointConditionalDistribution(
        TMLE.ConditionalDistribution(:T₁, (:T₂, :W₁)), 
        TMLE.ConditionalDistribution(:T₂, (:W₁,))
    )
    @test new_ĝ == ĝ
    # Update the collaborative strategy
    TMLE.update!(collaborative_strategy, new_g, new_ĝ)
    @test collaborative_strategy.remaining_confounders == Set{Symbol}([:W₂, :W₃])
    @test collaborative_strategy.current_confounders == Set{Symbol}([:W₁])
    # Let's iterate again
    step_k_candidate_iterator = TMLE.StepKPropensityScoreIterator(collaborative_strategy, Ψ, dataset, models, new_targeted_η̂ₙ)
    g_ĝ_candidates = collect(step_k_candidate_iterator)
    g_candidates = Set(first.(g_ĝ_candidates))
    @test g_candidates == Set([
        TMLE.JointConditionalDistribution(
            TMLE.ConditionalDistribution(:T₁, (:T₂, :W₁, :W₂)), 
            TMLE.ConditionalDistribution(:T₂, (:W₁,))
        ),
        TMLE.JointConditionalDistribution(
            TMLE.ConditionalDistribution(:T₁, (:T₂, :W₁)), 
            TMLE.ConditionalDistribution(:T₂, (:W₁, :W₃))
        )
    ])
    # Find optimal candidate again
    new_g, new_ĝ, new_targeted_η̂ₙ, new_loss, use_fluct = TMLE.step_k_best_candidate(
            collaborative_strategy,
            Ψ,
            dataset,
            models,
            fluctuation_model,
            targeted_η̂ₙ,
            loss;
            verbosity=verbosity,
            cache=cache,
            machine_cache=machine_cache,
    )
    @test new_g == TMLE.JointConditionalDistribution(
        TMLE.ConditionalDistribution(:T₁, (:T₂, :W₁, :W₂)), 
        TMLE.ConditionalDistribution(:T₂, (:W₁,))
    )
    
    # Full run: this leads to only W₃ being used for the propensity score
    ctmle = Tmle(collaborative_strategy=collaborative_strategy)
    Ψ = AIE(
        outcome = :Y,
        treatment_values = (
            T₁=(case=true, control=false),
            T₂=(case=true, control=false)
        ),
        treatment_confounders = [:W₁, :W₂, :W₃]
    )
    result_ctmle, cache = ctmle(Ψ, dataset;verbosity=0);
    targeted_η̂ = cache[:targeted_factors]
    @test targeted_η̂.treatments_factor[1].estimand == TMLE.ConditionalDistribution(:T₁, (:T₂, :W₃))
    @test targeted_η̂.treatments_factor[2].estimand == TMLE.ConditionalDistribution(:T₂, (:W₃,))
end

@testset "Integration Test using the AdaptiveCorrelationStrategy" begin
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
    collaborative_strategy = AdaptiveCorrelationStrategy()
    models = default_models(;
        Q_continuous = LinearRegressor(),
        G = LogisticClassifier(lambda=0.)
    )
    tmle = Tmle(;
        weighted=false,
        models=models,
        resampling=resampling,
        collaborative_strategy = collaborative_strategy,
        machine_cache=machine_cache,
    )
    # Initialize the relevant factors: no confounder is present in the propensity score
    η = TMLE.get_relevant_factors(Ψ, models, collaborative_strategy=collaborative_strategy)
    @test η.treatments_factor == TMLE.JointConditionalDistribution(
        TMLE.ConditionalDistribution(:T₁, (:T₂,)), 
        TMLE.ConditionalDistribution(:T₂, ())
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
    ps_T1_given_T2 = η̂ₙ.treatments_factor[1]
    fp = fitted_params(ps_T1_given_T2.machine)
    X, _ = ps_T1_given_T2.machine.data
    @test nrows(X) == n_samples
    fitted_variables = first.(fp.logistic_classifier.coefs)
    @test issubset(fitted_variables, [:T₂__false])
    ps_T2 = η̂ₙ.treatments_factor[2]
    fp = fitted_params(ps_T2.machine)
    @test haskey(fp, :target_distribution)
    @test ps_T2.machine.data[1] == DataFrame(INTERCEPT=fill(1., n_samples))
    
    ## One estimate for each estimand in the cache
    @test length(cache) == 4
    for estimand in (η, η.outcome_mean, η.treatments_factor...)
        @test length(cache[estimand]) == 1
    end
    # Collaborative Targeted Estimation
    estimator = TMLE.get_targeted_estimator(
        Ψ,
        tmle.collaborative_strategy,
        train_validation_indices,
        η̂ₙ;
        tol=tmle.tol,
        max_iter=tmle.max_iter,
        ps_lowerbound=tmle.ps_lowerbound,
        weighted=tmle.weighted,
        machine_cache=tmle.machine_cache,
        models=tmle.models
    )
    @test estimator isa TMLE.CMBasedCTMLE{TMLE.AdaptiveCorrelationStrategy}
    @test estimator.train_validation_indices == train_validation_indices
    fluctuation_model = estimator.fluctuation
    ## Check initialisation of the collaborative strategy
    TMLE.initialise!(estimator.collaborative_strategy, Ψ)
    @test estimator.collaborative_strategy.remaining_confounders == Set([:W₁, :W₃, :W₂])
    @test estimator.collaborative_strategy.current_confounders == Set()
    ## Check candidates initialisation: 
    ## - η corresponds to the propensity score with no confounders
    ## - The fluctuation is fitted on the whole dataset
    ## - A vector of (candidate, loss) is returned
    ## - A candidate is a targeted estimate
    targeted_η̂ₙ, loss = TMLE.get_initial_candidate(η, fluctuation_model, dataset;
        verbosity=verbosity,
        cache=cache,
        machine_cache=machine_cache
    )
    @test targeted_η̂ₙ isa TMLE.MLCMRelevantFactors
    @test targeted_η̂ₙ.outcome_mean.machine.model isa TMLE.Fluctuation
    @test nrows(targeted_η̂ₙ.outcome_mean.machine.data[1]) == n_samples
    @test targeted_η̂ₙ.treatments_factor === η̂ₙ.treatments_factor
    @test loss == TMLE.mean_loss(targeted_η̂ₙ, dataset)

    # Check CV candidates initialisation:
    ## - models are fitted on the training sets
    ## - The evaluation is made on the validation sets
    ## - The cache will contain all estimates to be reused
    cv_targeted_η̂ₙ, cv_loss = TMLE.get_initial_cv_candidate(η, dataset, fluctuation_model, train_validation_indices, models;
        cache=cache,
        verbosity=0,
        machine_cache=machine_cache
    );
    validation_losses = []
    for (fold_estimate, (train_indices, val_indices)) in zip(cv_targeted_η̂ₙ, train_validation_indices)
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
        for ps_component in fold_estimate.treatments_factor
            _, y_train = ps_component.machine.data
            @test y_train == dataset[!, ps_component.estimand.outcome][train_indices]
            @test ps_component in values(cache[ps_component.estimand])
        end
        # validation loss
        append!(validation_losses, TMLE.mean_loss(fold_estimate, selectrows(dataset, val_indices)))
    end
    @test mean(validation_losses) ≈ cv_loss

    # We now enter the main loop
    # A new propensity score and associated estimator is suggested
    new_g, new_ĝ, new_targeted_η̂ₙ, new_loss, use_fluct = TMLE.step_k_best_candidate(
            collaborative_strategy,
            Ψ,
            dataset,
            models,
            fluctuation_model,
            targeted_η̂ₙ,
            loss;
            verbosity=verbosity,
            machine_cache=machine_cache,
            cache=cache,
    )
    @test use_fluct == true
    @test new_g == TMLE.JointConditionalDistribution(
        TMLE.ConditionalDistribution(:T₁, (:T₂, :W₃)), 
        TMLE.ConditionalDistribution(:T₂, (:W₃,))
    )
    @test all(cde.train_validation_indices === nothing for cde in values(new_ĝ.cd_estimators))
    ## Check the propensity score estimate
    new_ĝₙ = new_targeted_η̂ₙ.treatments_factor
    for ps_component in new_ĝₙ
        @test nrows(ps_component.machine.data[1]) == n_samples
        @test ps_component in values(cache[ps_component.estimand])
    end
    ## We can check that the outcome model used is the targeted one
    outcome_mean_estimate_used = new_targeted_η̂ₙ.outcome_mean.machine.model.initial_factors.outcome_mean
    @test outcome_mean_estimate_used === targeted_η̂ₙ.outcome_mean
    ## The propensity score used for fluctuation is the new candidate
    fitted_propensity_score = new_targeted_η̂ₙ.outcome_mean.machine.model.initial_factors.treatments_factor.estimand
    @test fitted_propensity_score === new_g
    ## The loss should be smaller because we fluctuate through the previous model, however in finite samples
    ## I suppose this is not warranted, we check they are approximately equal and the loss with  Q̄n,k,* <  Q̄n,k
    @test loss ≈ new_loss atol=1e-5
    # We can pretend the step_k_best_candidate had used the non targeted outcome model
    new_targeted_η̂ₙ_bis, new_loss_bis = TMLE.get_new_targeted_candidate(targeted_η̂ₙ, new_ĝₙ, fluctuation_model, dataset;
            use_fluct=false,
            verbosity=verbosity,
            cache=cache,
            machine_cache=machine_cache
    )
    outcome_mean_estimate_used = new_targeted_η̂ₙ_bis.outcome_mean.machine.model.initial_factors.outcome_mean
    @test outcome_mean_estimate_used === targeted_η̂ₙ.outcome_mean.machine.model.initial_factors.outcome_mean
    ## Since this outcome mean model wasn't selected, the lost is bigger
    @test new_loss_bis > new_loss
    # The collaborative strategy is updated
    TMLE.update!(collaborative_strategy, new_g, new_ĝ)
    @test collaborative_strategy.remaining_confounders == Set([:W₁, :W₂])
    @test collaborative_strategy.current_confounders == Set([:W₃])

    # Evaluate the new candidate in CV passing through the fluctuated model
    new_cv_targeted_η̂ₙ, new_cv_loss = TMLE.evaluate_cv_candidate(cv_targeted_η̂ₙ, fluctuation_model, new_g, models, dataset, train_validation_indices; 
        use_fluct=true,
        verbosity=verbosity,
        cache=cache,
        machine_cache=machine_cache
    );
    for (fold_id, fold_candidate) in enumerate(new_cv_targeted_η̂ₙ)
        @test fold_candidate.outcome_mean.machine.model isa TMLE.Fluctuation
        @test fold_candidate.estimand.treatments_factor === new_g
        @test fold_candidate.outcome_mean.machine.model.initial_factors.outcome_mean === cv_targeted_η̂ₙ[fold_id].outcome_mean
    end
    @test TMLE.compute_validation_loss(new_cv_targeted_η̂ₙ, dataset, train_validation_indices) == new_cv_loss

    # Evaluate the new candidate in CV NOT passing through the fluctuated model
    new_cv_targeted_η̂ₙ_bis, new_cv_loss_bis = TMLE.evaluate_cv_candidate(cv_targeted_η̂ₙ, fluctuation_model, new_g, models, dataset, train_validation_indices; 
        use_fluct=false,
        verbosity=verbosity,
        cache=cache,
        machine_cache=machine_cache
    );
    for (fold_id, fold_candidate) in enumerate(new_cv_targeted_η̂ₙ_bis)
        @test fold_candidate.outcome_mean.machine.model isa TMLE.Fluctuation
        @test fold_candidate.estimand.treatments_factor === new_g
        @test fold_candidate.outcome_mean.machine.model.initial_factors.outcome_mean === cv_targeted_η̂ₙ[fold_id].outcome_mean.machine.model.initial_factors.outcome_mean
    end
    @test TMLE.compute_validation_loss(new_cv_targeted_η̂ₙ_bis, dataset, train_validation_indices) == new_cv_loss_bis

    # Check the full loop
    candidate_info = (targeted_η̂ₙ=targeted_η̂ₙ, loss=loss, cv_targeted_η̂ₙ=cv_targeted_η̂ₙ, cv_loss=cv_loss, id=1)
    best_candidate = TMLE.find_optimal_candidate(
        candidate_info, 
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
    @test best_candidate.cv_loss <= new_cv_loss
    @test best_candidate.cv_loss <= new_cv_loss_bis
end

end

true