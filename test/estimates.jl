module TestEstimates

using Test
using TMLE
using MLJBase
using MLJGLMInterface
using DataFrames

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
        ŷfold = predict(estimate.machines[foldid], dataset[val, expected_features])
        @test ŷ[val] == ŷfold
        @test μ̂[val] == mean.(ŷfold)
    end
    all(0. <= x <= 1. for x in TMLE.likelihood(estimate, dataset))
end

end

true