module TestGreedyStrategy

using Test
using TMLE
using MLJLinearModels
using MLJBase

TEST_DIR = joinpath(pkgdir(TMLE), "test")
include(joinpath(TEST_DIR, "counterfactual_mean_based", "interactions_simulations.jl"))

@testset "Test Interface" begin
    n_samples = 1_000
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
    η = TMLE.get_relevant_factors(Ψ, collaborative_strategy=collaborative_strategy)
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
        (TMLE.ConditionalDistribution(:T₁, (:T₂, :W₁)), TMLE.ConditionalDistribution(:T₂, (:W₁,))),
        (TMLE.ConditionalDistribution(:T₁, (:T₂,)), TMLE.ConditionalDistribution(:T₂, (:W₃,))),
        (TMLE.ConditionalDistribution(:T₁, (:T₂, :W₂)), TMLE.ConditionalDistribution(:T₂, ()))
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
    @test new_g == (TMLE.ConditionalDistribution(:T₁, (:T₂, :W₁)), TMLE.ConditionalDistribution(:T₂, (:W₁,)))
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
        (TMLE.ConditionalDistribution(:T₁, (:T₂, :W₁, :W₂)), TMLE.ConditionalDistribution(:T₂, (:W₁,))),
        (TMLE.ConditionalDistribution(:T₁, (:T₂, :W₁)), TMLE.ConditionalDistribution(:T₂, (:W₁, :W₃)))
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
    @test new_g == (TMLE.ConditionalDistribution(:T₁, (:T₂, :W₁, :W₂)), TMLE.ConditionalDistribution(:T₂, (:W₁,)))
end


end

true