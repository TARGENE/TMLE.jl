module TestComposition

using Test
using Random
using StableRNGs
using Distributions
using MLJLinearModels
using TMLE
using CategoricalArrays

function make_dataset(;n=100)
    rng = StableRNG(123)
    W = rand(rng, Uniform(), n)
    T = rand(rng, [0, 1], n)
    y = 3W .+ T .+ T.*W + rand(rng, Normal(0, 0.05), n)
    return (
        y = y,
        W = W,
        T = categorical(T)
    )
end

@testset "Test composition CM(1) - CM(0) = ATE(1,0)" begin
    dataset = make_dataset(;n=100)
    η_spec = (
        G = LogisticClassifier(lambda=0),
        Q = LinearRegressor()
    )
    # Conditional Mean T = 1
    CM₁ = CM(
        target = :y,
        treatment = (T=1,),
        confounders = [:W]
    )
    CM_result₁, _, cache = tmle(CM₁, η_spec, dataset, verbosity=0)
    # Conditional Mean T = 0
    CM₀ = CM(
        target = :y,
        treatment = (T=0,),
        confounders = [:W]
    )
    CM_result₀, _, cache = tmle!(cache, CM₀, verbosity=0)

    CM_result_composed = TMLE.compose(-, CM_result₁, CM_result₀)

    # Via ATE
    ATE₁₀ = ATE(
        target = :y,
        treatment = (T=(case=1, control=0),),
        confounders = [:W] 
    )
    ATE_result₁₀, _, cache = tmle!(cache, ATE₁₀, verbosity=0)
    @test TMLE.estimate(ATE_result₁₀) ≈ TMLE.estimate(CM_result_composed) atol = 1e-5
    var(ATE_result₁₀)
    var(CM_result_composed)
end

end

true