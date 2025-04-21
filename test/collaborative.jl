module TestCollaborative

using Test
using TMLE
using MLJLinearModels

TESTDIR = joinpath(pkgdir(TMLE), "test")
include(joinpath(TESTDIR, "counterfactual_mean_based", "interactions_simulations.jl"))

@testset "Integration Test GreedyCollaboration" begin
    dataset, Ψ₀ = continuous_outcome_binary_treatment_pb(n=10_000)
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
    verbosity = 2
    tmle = TMLEE(;collaborative_strategy=GreedyCollaboration())
    # Initialize the relevant factors' estimates
    relevant_factors = TMLE.get_relevant_factors(Ψ)
    initial_factors_estimator = TMLE.CMRelevantFactorsEstimator(tmle.resampling, tmle.collaborative_strategy, tmle.models)
    initial_factors_estimate = initial_factors_estimator(relevant_factors, dataset; 
        cache=cache, 
        verbosity=verbosity, 
        machine_cache=tmle.machine_cache
    )
    initial_factors_estimate.propensity_score[1] == TMLE.ConditionalDistribution(:T₁, (:T₂,))
    initial_factors_estimate.propensity_score[2] == TMLE.ConditionalDistribution(:T₂, ())

    propensity_score = TMLE.initialise_propensity_score(collaborative_strategy, Ψ)
    @test propensity_score == (
        TMLE.ConditionalDistribution(:T₁, (:T₂,)),
        TMLE.ConditionalDistribution(:T₂, ())
    )

    model = LogisticClassifier()
    ps_estimator = TMLE.initialise_propensity_score_estimator(collaborative_strategy, model)
    @test ps_estimator == TMLE.MLConditionalDistributionEstimator(model)



end

end

true