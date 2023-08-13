module TestComposition

using Test
using Random
using StableRNGs
using Distributions
using MLJLinearModels
using TMLE
using CategoricalArrays

function make_dataset_and_scm(;n=100)
    rng = StableRNG(123)
    W = rand(rng, Uniform(), n)
    T = rand(rng, [0, 1], n)
    Y = 3W .+ T .+ T.*W + rand(rng, Normal(0, 0.05), n)
    dataset =  (
        Y = Y,
        W = W,
        T = categorical(T)
    )
    scm = StaticConfoundedModel(:Y, :T, :W)
    return dataset, scm
end

@testset "Test cov" begin
    n = 10
    X = rand(n, 2)
    ER₁ = TMLE.TMLEstimate(1., X[:, 1])
    ER₂ = TMLE.TMLEstimate(0., X[:, 2])
    Σ = cov(ER₁, ER₂)
    @test size(Σ) == (2, 2)
    @test Σ == cov(X) 
end

@testset "Test composition CM(1) - CM(0) = ATE(1,0)" begin
    dataset, scm = make_dataset_and_scm(;n=1000)
    # Conditional Mean T = 1
    CM₁ = CM(
        scm,
        outcome = :Y,
        treatment = (T=1,)
    )
    CM_result₁, _ = tmle!(CM₁, dataset, verbosity=0)
    # Conditional Mean T = 0
    CM₀ = CM(
        scm,
        outcome = :Y,
        treatment = (T=0,)
    )
    CM_result₀, _ = tmle!(CM₀, dataset, verbosity=0)
    # Composition of TMLE

    CM_result_composed_tmle = compose(-, tmle(CM_result₁), tmle(CM_result₀));
    CM_result_composed_ose = compose(-, ose(CM_result₁), ose(CM_result₀));

    # Via ATE
    ATE₁₀ = ATE(
        scm,
        outcome = :Y,
        treatment = (T=(case=1, control=0),),
    )
    ATE_result₁₀, _ = tmle!(ATE₁₀, dataset, verbosity=0)
    # Check composed TMLE
    @test estimate(tmle(ATE_result₁₀)) ≈ estimate(CM_result_composed_tmle) atol = 1e-7
    # T Test
    composed_confint = collect(confint(OneSampleTTest(CM_result_composed_tmle)))
    tmle_confint = collect(confint(OneSampleTTest(tmle(ATE_result₁₀))))
    @test tmle_confint ≈ composed_confint atol=1e-4
    # Z Test
    composed_confint = collect(confint(OneSampleZTest(CM_result_composed_tmle)))
    tmle_confint = collect(confint(OneSampleZTest(tmle(ATE_result₁₀))))
    @test tmle_confint ≈ composed_confint atol=1e-4
    # Variance
    @test var(tmle(ATE_result₁₀)) ≈ var(CM_result_composed_tmle) atol = 1e-3

    # Check composed OSE
    @test estimate(ose(ATE_result₁₀)) ≈ estimate(CM_result_composed_ose) atol = 1e-7
    # T Test
    composed_confint = collect(confint(OneSampleTTest(CM_result_composed_ose)))
    ose_confint = collect(confint(OneSampleTTest(ose(ATE_result₁₀))))
    @test ose_confint ≈ composed_confint atol=1e-4
    # Z Test
    composed_confint = collect(confint(OneSampleZTest(CM_result_composed_tmle)))
    ose_confint = collect(confint(OneSampleZTest(ose(ATE_result₁₀))))
    @test ose_confint ≈ composed_confint atol=1e-4
    # Variance
    @test var(ose(ATE_result₁₀)) ≈ var(CM_result_composed_ose) atol = 1e-3
end

@testset "Test compose multidimensional function" begin
    dataset, scm = make_dataset_and_scm(;n=1000)
    CM₁ = CM(
        scm,
        outcome = :Y,
        treatment = (T=1,)
    )
    CM_result₁, _ = tmle!(CM₁, dataset, verbosity=0)

    CM₀ = CM(
        scm,
        outcome = :Y,
        treatment = (T=0,)
    )
    CM_result₀, _ = tmle!(CM₀, dataset, verbosity=0)
    f(x, y) = [x^2 - y, x/y, 2x + 3y]
    CM_result_composed = compose(f, CM_result₁.tmle, CM_result₀.tmle)

    @test estimate(CM_result_composed) == f(estimate(CM_result₁.tmle), TMLE.estimate(CM_result₀.tmle))
    @test size(var(CM_result_composed)) == (3, 3)
end

end

true