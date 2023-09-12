module TestEstimates

using Test
using TMLE
using MLJBase
using MLJGLMInterface
using DataFrames
using Distributions
using LogExpFunctions

scm = SCM(
    SE(:Y, [:T, :W]),
    SE(:T₁, [:W]),
    SE(:T₂, [:W])
)

@testset "Test MLConditionalDistribution" begin
    X, y = make_regression(100, 4)
    dataset = DataFrame(Y=y, T₁=X.x1, T₂=X.x2, W=X.x3)
    estimand = ConditionalDistribution(scm, :Y, [:W, :T₁, :T₂])
    train_validation_indices = nothing
    model = LinearRegressor()
    estimate = TMLE.estimate(
        estimand, dataset, model, train_validation_indices;
        factors_cache=nothing,
        verbosity=0
    )
    @test estimate isa TMLE.MLConditionalDistribution
    expected_features = [:T₁, :T₂, :W]
    @test fitted_params(estimate.machine).features == expected_features
    ŷ = predict(estimate, dataset)
    @test ŷ == predict(estimate.machine, dataset[!, expected_features])
    μ̂ = TMLE.expected_value(estimate, dataset)
    @test μ̂ == mean.(ŷ)
    @test all(0. <= x <= 1. for x in TMLE.likelihood(estimate, dataset))
end

@testset "Test SampleSplitMLConditionalDistribution" begin
    n = 100
    nfolds = 3
    X, y = make_regression(n, 4)
    dataset = DataFrame(Y=y, T₁=X.x1, T₂=X.x2, W=X.x3)
    estimand = ConditionalDistribution(scm, :Y, [:W, :T₁, :T₂])
    train_validation_indices = collect(MLJBase.train_test_pairs(CV(nfolds=nfolds), 1:n, dataset))
    model = LinearRegressor()
    estimate = TMLE.estimate(
        estimand, dataset, model, train_validation_indices;
        factors_cache=nothing,
        verbosity=0
    )
    @test estimate isa TMLE.SampleSplitMLConditionalDistribution
    expected_features = [:T₁, :T₂, :W]
    @test all(fitted_params(mach).features == expected_features for mach in estimate.machines)
    ŷ = predict(estimate, dataset)
    μ̂ = TMLE.expected_value(estimate, dataset)
    for foldid in 1:nfolds
        train, val = train_validation_indices[foldid]
        # The predictions on validation samples are made from
        # the machine trained on the train sample
        ŷfold = predict(estimate.machines[foldid], dataset[val, expected_features])
        @test ŷ[val] == ŷfold
        @test μ̂[val] == mean.(ŷfold)
    end
    all(0. <= x <= 1. for x in TMLE.likelihood(estimate, dataset))
end

@testset "Test MLCMRelevantFactors" begin
    n = 100
    W = rand(Normal(), n)
    T₁ = rand(n) .< logistic.(1 .- W)
    Y = T₁ .+ W .+ rand(n)
    dataset = DataFrame(Y=Y, W=W, T₁=categorical(T₁, ordered=true))
    models = (Y=with_encoder(LinearRegressor()), T₁=LinearBinaryClassifier())
    estimand = TMLE.CMRelevantFactors(
        scm,
        outcome_mean = TMLE.ConditionalDistribution(scm, :Y, Set([:T₁, :W])),
        propensity_score = (TMLE.ConditionalDistribution(scm, :T₁, Set([:W])),)
    )
    # No resampling
    resampling = nothing
    estimate = Distributions.estimate(estimand, resampling, models, dataset; 
        factors_cache=nothing, 
        verbosity=0
    )
    outcome_model_features = [:T₁, :W]
    @test predict(estimate.outcome_mean, dataset) == predict(estimate.outcome_mean.machine, dataset[!, outcome_model_features])
    @test predict(estimate.propensity_score[1], dataset).prob_given_ref == predict(estimate.propensity_score[1].machine, dataset[!, [:W]]).prob_given_ref

    # No resampling
    nfolds = 3
    resampling = CV(nfolds=nfolds)
    estimate = Distributions.estimate(estimand, resampling, models, dataset; 
        factors_cache=nothing, 
        verbosity=0
    )
    train_validation_indices = estimate.outcome_mean.train_validation_indices
    # Check all models share the same train/validation indices
    @test train_validation_indices == estimate.propensity_score[1].train_validation_indices
    outcome_model_features = [:T₁, :W]
    ŷ = predict(estimate.outcome_mean, dataset)
    t̂ = predict(estimate.propensity_score[1], dataset)
    for foldid in 1:nfolds
        train, val = train_validation_indices[foldid]
        # The predictions on validation samples are made from
        # the machine trained on the train sample
        ŷfold = predict(estimate.outcome_mean.machines[foldid], dataset[val, outcome_model_features])
        @test ŷ[val] == ŷfold
        t̂fold = predict(estimate.propensity_score[1].machines[foldid], dataset[val, [:W]])
        @test t̂fold.prob_given_ref == t̂[val].prob_given_ref
    end

end


end

true