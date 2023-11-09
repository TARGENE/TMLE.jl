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
    Y = 3W .+ T .+ T.*W + rand(rng, Normal(0, 0.05), n)
    dataset =  (
        Y = Y,
        W = W,
        T = categorical(T)
    )
    return dataset
end

@testset "Test cov" begin
    Ψ = CM(
        outcome = :Y,
        treatment_values = (T=1,),
        treatment_confounders = (T=[:W],)
    )
    n = 10
    X = rand(n, 2)
    ER₁ = TMLE.TMLEstimate(Ψ, 1., X[:, 1])
    ER₂ = TMLE.TMLEstimate(Ψ, 0., X[:, 2])
    Σ = cov(ER₁, ER₂)
    @test size(Σ) == (2, 2)
    @test Σ == cov(X) 
end

@testset "Test composition CM(1) - CM(0) = ATE(1,0)" begin
    dataset = make_dataset(;n=1000)
    # Counterfactual Mean T = 1
    CM₁ = CM(
        outcome = :Y,
        treatment_values = (T=1,),
        treatment_confounders = (T=[:W],)
    )
    models = (
        Y = with_encoder(LinearRegressor()),
        T = LogisticClassifier(lambda=0)
    )
    tmle = TMLEE(models=models)
    ose = OSE(models=models)
    cache = Dict()

    CM_tmle_result₁, cache = tmle(CM₁, dataset; cache=cache, verbosity=0)
    CM_ose_result₁, cache = ose(CM₁, dataset; cache=cache, verbosity=0)
    # Counterfactual Mean T = 0
    CM₀ = CM(
        outcome = :Y,
        treatment_values = (T=0,),
        treatment_confounders = (T=[:W],)
    )
    CM_tmle_result₀, cache = tmle(CM₀, dataset; cache=cache, verbosity=0)
    CM_ose_result₀, cache = ose(CM₀, dataset; cache=cache, verbosity=0)
    # Composition of TMLE

    CM_result_composed_tmle = compose(-, CM_tmle_result₁, CM_tmle_result₀);
    CM_result_composed_ose = compose(-, CM_ose_result₁, CM_ose_result₀);

    # Via ATE
    ATE₁₀ = ATE(
        outcome = :Y,
        treatment_values = (T=(case=1, control=0),),
        treatment_confounders = (T=[:W],)
    )
    # Check composed TMLE
    ATE_tmle_result₁₀, cache = tmle(ATE₁₀, dataset; cache=cache, verbosity=0)
    @test estimate(ATE_tmle_result₁₀) ≈ estimate(CM_result_composed_tmle) atol = 1e-7
    # T Test
    composed_confint = collect(confint(OneSampleTTest(CM_result_composed_tmle)))
    tmle_confint = collect(confint(OneSampleTTest(ATE_tmle_result₁₀)))
    @test tmle_confint ≈ composed_confint atol=1e-4
    # Z Test
    composed_confint = collect(confint(OneSampleZTest(CM_result_composed_tmle)))
    tmle_confint = collect(confint(OneSampleZTest(ATE_tmle_result₁₀)))
    @test tmle_confint ≈ composed_confint atol=1e-4
    # Variance
    @test var(ATE_tmle_result₁₀) ≈ var(CM_result_composed_tmle) atol = 1e-3

    # Check composed OSE
    ATE_ose_result₁₀, cache = ose(ATE₁₀, dataset; cache=cache, verbosity=0)
    @test estimate(ATE_ose_result₁₀) ≈ estimate(CM_result_composed_ose) atol = 1e-7
    # T Test
    composed_confint = collect(confint(OneSampleTTest(CM_result_composed_ose)))
    ose_confint = collect(confint(OneSampleTTest(ATE_ose_result₁₀)))
    @test ose_confint ≈ composed_confint atol=1e-4
    # Z Test
    composed_confint = collect(confint(OneSampleZTest(CM_result_composed_ose)))
    ose_confint = collect(confint(OneSampleZTest(ATE_ose_result₁₀)))
    @test ose_confint ≈ composed_confint atol=1e-4
    # Variance
    @test var(ATE_ose_result₁₀) ≈ var(CM_result_composed_ose) atol = 1e-3
end

@testset "Test compose multidimensional function" begin
    dataset = make_dataset(;n=1000)
    models = (
        Y = with_encoder(LinearRegressor()),
        T = LogisticClassifier(lambda=0)
    )
    tmle = TMLEE(models=models)
    cache = Dict()
    
    CM₁ = CM(
        outcome = :Y,
        treatment_values = (T=1,),
        treatment_confounders = (T=[:W],)
    )
    CM_result₁, cache = tmle(CM₁, dataset; cache=cache, verbosity=0)

    CM₀ = CM(
        outcome = :Y,
        treatment_values = (T=0,),
        treatment_confounders = (T=[:W],)
    )
    CM_result₀, cache = tmle(CM₀, dataset; cache=cache, verbosity=0)
    f(x, y) = [x^2 - y, x/y, 2x + 3y]
    CM_result_composed = compose(f, CM_result₁, CM_result₀)

    @test estimate(CM_result_composed) == f(estimate(CM_result₁), TMLE.estimate(CM_result₀))
    @test size(var(CM_result_composed)) == (3, 3)
end

end

true