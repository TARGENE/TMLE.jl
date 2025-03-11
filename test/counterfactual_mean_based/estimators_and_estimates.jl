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
using MLJBase
using CSV

DATADIR = joinpath(pkgdir(TMLE), "test", "data")

function make_dataset()
    n = 100
    W = rand(Normal(), n)
    T₁ = rand(n) .< logistic.(1 .- W)
    Y = T₁ .+ W .+ rand(n)
    dataset = DataFrame(Y=Y, W=W, T₁=categorical(T₁, ordered=true))
    return dataset
end

@testset "Test CMRelevantFactorsEstimator" begin
    dataset = make_dataset()
    # Estimand
    Q = TMLE.ConditionalDistribution(:Y, [:T₁, :W])
    G = (TMLE.ConditionalDistribution(:T₁, [:W]),)
    η = TMLE.CMRelevantFactors(outcome_mean=Q, propensity_score=G)
    # Estimator
    models = Dict(
        :Y  => with_encoder(LinearRegressor()), 
        :T₁ => LogisticClassifier()
    )
    η̂ = TMLE.CMRelevantFactorsEstimator(models=models)
    # Estimate
    fit_log = (
        (:info, string("Required ", TMLE.string_repr(η))),
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
    new_models = Dict(
        :Y  => with_encoder(LinearRegressor()), 
        :T₁ => LogisticClassifier()
    )
    new_η̂ = TMLE.CMRelevantFactorsEstimator(models=new_models)
    @test TMLE.key(η, new_η̂) == TMLE.key(η, η̂)
    full_reuse_log = (:info, TMLE.reuse_string(η))
    @test_logs full_reuse_log new_η̂(η, dataset; cache=cache, verbosity=1)
    # Changing one model, only the other one is refitted
    new_models = Dict(
        :Y  => with_encoder(LinearRegressor()), 
        :T₁ => LogisticClassifier(fit_intercept=false)
    )
    new_η̂ = TMLE.CMRelevantFactorsEstimator(models=new_models)
    @test TMLE.key(η, new_η̂) != TMLE.key(η, η̂)
    partial_reuse_log = (
        (:info, string("Required ", TMLE.string_repr(η))),
        (:info, TMLE.fit_string(G[1])),
        (:info, TMLE.reuse_string(Q))
    )
    @test_logs partial_reuse_log... new_η̂(η, dataset; cache=cache, verbosity=1)

    # Adding a resampling strategy
    resampled_η̂ = TMLE.CMRelevantFactorsEstimator(models=new_models, resampling=CV(nfolds=3))
    @test TMLE.key(η, new_η̂) != TMLE.key(η, resampled_η̂)
    η̂ₙ = @test_logs fit_log... resampled_η̂(η, dataset; cache=cache, verbosity=1)
    @test length(η̂ₙ.outcome_mean.machines) == 3
    @test length(η̂ₙ.propensity_score[1].machines) == 3
    @test η̂ₙ.outcome_mean.train_validation_indices == η̂ₙ.propensity_score[1].train_validation_indices
end

@testset "Test FitFailedError" begin
    dataset = make_dataset()
    # Estimand
    Q = TMLE.ConditionalDistribution(:Y, [:T₁, :W])
    G = (TMLE.ConditionalDistribution(:T₁, [:W]),)
    η = TMLE.CMRelevantFactors(outcome_mean=Q, propensity_score=G)
    # Propensity score model is ill-defined
    models = Dict(
        :Y  => with_encoder(LinearRegressor()), 
        :T₁ => LinearRegressor()
    )
    η̂ = TMLE.CMRelevantFactorsEstimator(models=models)
    try 
        η̂(η, dataset; verbosity=0)
        @test true === false
    catch e
        @test e isa TMLE.FitFailedError
        @test e.model isa LinearRegressor
        @test e.msg == TMLE.propensity_score_fit_error_msg(G[1])
    end
    # Outcome Mean model is ill-defined
    models = Dict(
        :Y  => LogisticClassifier(), 
        :T₁ => LogisticClassifier(fit_intercept=false)
    )
    η̂ = TMLE.CMRelevantFactorsEstimator(models=models)
    try 
        η̂(η, dataset; verbosity=0)
        @test true === false
    catch e
        @test e isa TMLE.FitFailedError
        @test e.model isa LogisticClassifier
        @test e.msg == TMLE.outcome_mean_fit_error_msg(Q)
    end
    # Fluctuation Pos Def Exception
    pos_def_error_dataset = CSV.read(joinpath(DATADIR, "posdef_error_dataset.csv"), DataFrame)
    outcome = Symbol("G25 Other extrapyramidal and movement disorders")
    treatment = Symbol("2:14983:G:A")
    pos_def_error_dataset[!, treatment] = categorical(pos_def_error_dataset[!, treatment])
    pos_def_error_dataset[!, outcome] = categorical(pos_def_error_dataset[!, outcome])
    Ψ = ATE(
        outcome=outcome, 
        treatment_values = NamedTuple{(treatment,)}([(case = "GG", control = "AG")]), 
        treatment_confounders = (:PC1, :PC2, :PC3, :PC4, :PC5, :PC6)
    )
    Q = TMLE.ConditionalDistribution(outcome, [treatment, :PC1, :PC2, :PC3, :PC4, :PC5, :PC6])
    tmle = TMLEE(models=TMLE.default_models(Q_binary=LogisticClassifier(), G = LogisticClassifier()))
    try 
        tmle(Ψ, pos_def_error_dataset)
        @test true === false
    catch e
        @test e isa TMLE.FitFailedError
        @test e.model isa TMLE.Fluctuation
        @test e.msg == TMLE.outcome_mean_fluctuation_fit_error_msg(Q)
    end
end

@testset "Test structs are concrete types" begin
    for type in (OSE, TMLEE, NAIVE)
        @test isconcretetype(type)
    end
end

end

true