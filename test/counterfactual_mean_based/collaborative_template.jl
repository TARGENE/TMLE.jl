module TestCollaborative

using Test
using TMLE
using MLJLinearModels
using MLJBase
using Distributions
using StatisticalMeasures

TEST_DIR = joinpath(pkgdir(TMLE), "test")
include(joinpath(TEST_DIR, "counterfactual_mean_based", "interactions_simulations.jl"))

@testset "Test compute_loss" begin
    dataset, Ψ₀ = continuous_outcome_binary_treatment_pb(n=1_000)
    # With a continuous outcome
    ## With a deterministic model: MLJLinearModels.LinearRegressor
    estimator = TMLE.MLConditionalDistributionEstimator(LinearRegressor(), nothing)
    cde = estimator(
        TMLE.ConditionalDistribution(:Y, (:W₁, :W₂)),
        dataset,
    )
    loss = TMLE.compute_loss(cde, dataset)
    ŷ = TMLE.predict(cde, dataset)
    @test loss ≈ abs.(ŷ .- dataset.Y)
    ## With a probabilistic model: TMLE.LinearRegressor results in GLM model
    estimator = TMLE.MLConditionalDistributionEstimator(TMLE.LinearRegressor(), nothing)
    cde = estimator(
        TMLE.ConditionalDistribution(:Y, (:W₁, :W₂)),
        dataset,
    )
    ŷ = TMLE.predict(cde, dataset)
    @test ŷ isa Vector{Normal{Float64}}
    new_loss = TMLE.compute_loss(cde, dataset)
    @test new_loss ≈ loss
    # With a binary outcome
    estimator = TMLE.MLConditionalDistributionEstimator(LogisticClassifier(), nothing)
    cde = estimator(
        TMLE.ConditionalDistribution(:T₁, (:W₁, :W₂)),
        dataset,
    )
    ŷ = TMLE.predict(cde, dataset)
    @test ŷ isa UnivariateFiniteVector
    loss = TMLE.compute_loss(cde, dataset)
    @test loss ≈ measurements(LogLoss(), ŷ, dataset.T₁)
end

end

true