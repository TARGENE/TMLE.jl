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

basicCM(val) = CM(
    target = :y,
    treatment = (T=val,),
    confounders = [:W]
)

@testset "Test cov" begin
    n = 10
    X = rand(n, 2)
    ER₁ = TMLE.PointTMLE(1., X[:, 1])
    ER₂ = TMLE.PointTMLE(0., X[:, 2])
    Σ = cov(ER₁, ER₂)
    @test size(Σ) == (2, 2)
    @test Σ == cov(X) 
end

@testset "Test composition CM(1) - CM(0) = ATE(1,0)" begin
    dataset = make_dataset(;n=1000)
    η_spec = (
        G = LogisticClassifier(lambda=0),
        Q = LinearRegressor()
    )
    # Conditional Mean T = 1
    CM_result₁, _, cache = tmle(basicCM(1), η_spec, dataset, verbosity=0)
    # Conditional Mean T = 0
    CM_result₀, _, cache = tmle!(cache, basicCM(0), verbosity=0)
    # Via Composition
    CM_result_composed = compose(-, CM_result₁, CM_result₀)

    # Via ATE
    ATE₁₀ = ATE(
        target = :y,
        treatment = (T=(case=1, control=0),),
        confounders = [:W] 
    )
    ATE_result₁₀, _, cache = tmle!(cache, ATE₁₀, verbosity=0)
    @test TMLE.estimate(ATE_result₁₀) ≈ TMLE.estimate(CM_result_composed) atol = 1e-7
    @test var(ATE_result₁₀) ≈ var(CM_result_composed) atol = 1e-7
end

@testset "Test compose multidimensional function" begin
    dataset = make_dataset(;n=1000)
    η_spec = (
        G = LogisticClassifier(lambda=0),
        Q = LinearRegressor()
    )
    CM_result₁, _, cache = tmle(basicCM(1), η_spec, dataset, verbosity=0)
    CM_result₀, _, cache = tmle!(cache, basicCM(0), verbosity=0)
    f(x, y) = [x^2 - y, x/y, 2x + 3y]
    CM_result_composed = compose(f, CM_result₁, CM_result₀)

    @test TMLE.estimate(CM_result_composed) == f(TMLE.estimate(CM_result₁), TMLE.estimate(CM_result₀))
    @test size(var(CM_result_composed)) == (3, 3)
end

end

true