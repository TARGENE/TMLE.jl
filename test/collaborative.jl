module TestCollaborative

using Test
using TMLE
using MLJLinearModels

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
    verbosity = 1
    collaborative_strategy = AdaptiveCorrelationOrdering(resampling=StratifiedCV())
    tmle = TMLEE(;collaborative_strategy=collaborative_strategy)
    # Initialize the relevant factors' estimates
    η = TMLE.get_relevant_factors(Ψ)
    initial_factors_estimator = TMLE.CMRelevantFactorsEstimator(tmle.resampling, tmle.collaborative_strategy, tmle.models)
    η̂ₙ = initial_factors_estimator(η, dataset; 
        cache=cache, 
        verbosity=verbosity, 
        machine_cache=tmle.machine_cache
    )
    
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

    candidates = estimator(η, dataset; 
        cache=cache, 
        verbosity=verbosity,
        machine_cache=tmle.machine_cache
    )

    getfield.(candidates, :loss)

end

end

true