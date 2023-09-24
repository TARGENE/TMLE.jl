module TestEstimation

using Test
using Tables
using StableRNGs
using TMLE
using DataFrames
using Distributions
using MLJLinearModels
using CategoricalArrays
using LogExpFunctions

include(joinpath(dirname(@__DIR__), "helper_fns.jl"))

@testset "Test CMRelevantFactorsEstimator" begin
    n = 100
    W = rand(Normal(), n)
    T₁ = rand(n) .< logistic.(1 .- W)
    Y = T₁ .+ W .+ rand(n)
    dataset = DataFrame(Y=Y, W=W, T₁=categorical(T₁, ordered=true))
    # Estimand
    Q = TMLE.ConditionalDistribution(:Y, [:T₁, :W])
    G = (TMLE.ConditionalDistribution(:T₁, [:W]),)
    η = TMLE.CMRelevantFactors(outcome_mean=Q, propensity_score=G)
    # Estimator
    models = (
        Y = with_encoder(LinearRegressor()), 
        T₁ = LogisticClassifier()
    )
    η̂ = TMLE.CMRelevantFactorsEstimator(models=models)
    # Estimate
    fit_log = (
        (:info, TMLE.fit_string(η)),
        (:info, TMLE.fit_string(G[1])),
        (:info, TMLE.fit_string(Q))
    )
    cache = Dict()
    η̂ₙ = @test_logs fit_log... η̂(η, dataset; cache=cache, verbosity=1)
    # Test both sub estimands have been fitted
    @test η̂ₙ.outcome_mean isa TMLE.MLConditionalDistribution
    @test fitted_params(η̂ₙ.outcome_mean.machine) isa NamedTuple
    @test η̂ₙ.propensity_score[1] isa TMLE.MLConditionalDistribution
    @test fitted_params(η̂ₙ.propensity_score[1].machine) isa NamedTuple

    # Both models unchanged, η̂ₙ is fully reused
    new_models = (
        Y = with_encoder(LinearRegressor()), 
        T₁ = LogisticClassifier()
    )
    new_η̂ = TMLE.CMRelevantFactorsEstimator(models=new_models)
    @test TMLE.key(η, new_η̂) == TMLE.key(η, η̂)
    full_reuse_log = (:info, TMLE.reuse_string(η))
    @test_logs full_reuse_log new_η̂(η, dataset; cache=cache, verbosity=1)

    # Changing one model, only the other one is refitted
    new_models = (
        Y = with_encoder(LinearRegressor()), 
        T₁ = LogisticClassifier(fit_intercept=false)
    )
    new_η̂ = TMLE.CMRelevantFactorsEstimator(models=new_models)
    @test TMLE.key(η, new_η̂) != TMLE.key(η, η̂)
    partial_reuse_log = (
        (:info, TMLE.fit_string(η)),
        (:info, TMLE.fit_string(G[1])),
        (:info, TMLE.reuse_string(Q))
    )
    @test_logs partial_reuse_log... new_η̂(η, dataset; cache=cache, verbosity=1)

end


end

true