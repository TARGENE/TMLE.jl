module TestComposition

using Test
using Random
using StableRNGs
using Distributions
using MLJLinearModels
using TMLE
using CategoricalArrays
using LogExpFunctions
using HypothesisTests

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
    ER₁ = TMLE.TMLEstimate(Ψ, 1., 1., n, X[:, 1])
    ER₂ = TMLE.TMLEstimate(Ψ, 0., 1., n, X[:, 2])
    Σ = TMLE.covariance_matrix(ER₁, ER₂)
    @test size(Σ) == (2, 2)
    @test Σ == cov(X) 
end

@testset "Test to_dict and from_dict! ComposedEstimand" begin
    ATE₁ = ATE(
        outcome=:Y,
        treatment_values = (T=(case=1, control=0),),
        treatment_confounders = (T=[:W],)
    )
    ATE₂ = ATE(
        outcome=:Y,
        treatment_values = (T=(case=2, control=1),),
        treatment_confounders = (T=[:W],)
    )
    diff = ComposedEstimand(-, (ATE₁, ATE₂))
    d = TMLE.to_dict(diff)
    diff_from_dict = TMLE.from_dict!(d)
    @test diff_from_dict == diff

    # Anonymous function will raise
    anonymousdiff = ComposedEstimand((x,y) -> x - y, (ATE₁, ATE₂))
    @test_throws ArgumentError TMLE.to_dict(anonymousdiff)
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
    # Estimate Individually
    CM_tmle_result₀, cache = tmle(CM₀, dataset; cache=cache, verbosity=0)
    CM_ose_result₀, cache = ose(CM₀, dataset; cache=cache, verbosity=0)
    # Compose estimates
    CM_result_composed_tmle = compose(-, CM_tmle_result₁, CM_tmle_result₀);
    CM_result_composed_ose = compose(-, CM_ose_result₁, CM_ose_result₀);
    # Estimate via ComposedEstimand
    composed_estimand = ComposedEstimand(-, (CM₁, CM₀))
    composed_estimate, cache = tmle(composed_estimand, dataset; cache=cache, verbosity=0)
    @test composed_estimate.estimand == CM_result_composed_tmle.estimand
    @test CM_result_composed_tmle.estimate == composed_estimate.estimate
    @test CM_result_composed_tmle.cov == composed_estimate.cov
    @test CM_result_composed_tmle.n == composed_estimate.n
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

@testset "Test Joint Interaction" begin
    # Dataset
    n = 100
    rng = StableRNG(123)

    W = rand(rng, n)

    θT₁ = rand(rng, Normal(), 3)
    pT₁ =  softmax(W*θT₁', dims=2)
    T₁ = [rand(rng, Categorical(collect(p))) for p in eachrow(pT₁)]
    
    θT₂ = rand(rng, Normal(), 3)
    pT₂ =  softmax(W*θT₂', dims=2)
    T₂ = [rand(rng, Categorical(collect(p))) for p in eachrow(pT₂)]

    Y = 1 .+ W .+ T₁ .- T₂ .- T₁.*T₂ .+ rand(rng, Normal())
    dataset = (
        W = W,
        T₁ = categorical(T₁),
        T₂ = categorical(T₂),
        Y = Y
    )
    IATE₁ = IATE(
        outcome = :Y,
        treatment_values = (T₁=(case=2, control=1), T₂=(case=2, control=1)),
        treatment_confounders = (T₁ = [:W], T₂ = [:W])
    )
    IATE₂ = IATE(
        outcome = :Y,
        treatment_values = (T₁=(case=3, control=1), T₂=(case=3, control=1)),
        treatment_confounders = (T₁ = [:W], T₂ = [:W])
    )
    IATE₃ = IATE(
        outcome = :Y,
        treatment_values = (T₁=(case=3, control=2), T₂=(case=3, control=2)),
        treatment_confounders = (T₁ = [:W], T₂ = [:W])
    )
    jointIATE = ComposedEstimand(TMLE.joint_estimand, (IATE₁, IATE₂, IATE₃))
    ose = OSE(models=TMLE.default_models(G=LogisticClassifier(), Q_continuous=LinearRegressor()))
    jointEstimate, _ = ose(jointIATE, dataset, verbosity=0)

    testres = significance_test(jointEstimate)
    @test testres.x̄ ≈ jointEstimate.estimate
    @test pvalue(testres) < 1e-10

    emptied_estimate = TMLE.emptyIC(jointEstimate)
    for Ψ̂ₙ in emptied_estimate.estimates
        @test Ψ̂ₙ.IC == []
    end

    pval_threshold = 1e-3
    maybe_emptied_estimate = TMLE.emptyIC(jointEstimate, pval_threshold=pval_threshold)
    n_empty = 0
    for i in 1:3
        pval = pvalue(significance_test(jointEstimate.estimates[i]))
        maybe_emptied_IC = maybe_emptied_estimate.estimates[i].IC
        if pval > pval_threshold
            @test maybe_emptied_IC == []
            n_empty += 1
        else
            @test length(maybe_emptied_IC) == n
        end
    end
    @test n_empty > 0

    d = TMLE.to_dict(jointEstimate)
    jointEstimate_fromdict = TMLE.from_dict!(d)

    @test jointEstimate.estimand == jointEstimate_fromdict.estimand
    @test jointEstimate.cov == jointEstimate_fromdict.cov
    @test jointEstimate.estimate == jointEstimate_fromdict.estimate
    @test jointEstimate.n == jointEstimate_fromdict.n
    @test length(jointEstimate_fromdict.estimates) == 3

    if VERSION >= v"1.9"
        using JSON
        filename, _ = mktemp()
        TMLE.write_json(filename, jointEstimate)
        from_json = TMLE.read_json(filename, use_mmap=false)
        @test jointEstimate.estimand == from_json.estimand
        @test jointEstimate.cov == from_json.cov
        @test jointEstimate.estimate == from_json.estimate
        @test jointEstimate.n == from_json.n
        @test length(jointEstimate_fromdict.estimates) == 3
    end
end


end

true