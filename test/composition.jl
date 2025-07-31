module TestComposition

using Test
using TMLE
using DifferentiationInterface
using Zygote
using Random
using StableRNGs
using Distributions
using MLJLinearModels
using CategoricalArrays
using LogExpFunctions
using HypothesisTests
using DataFrames

function make_dataset(;n=100)
    rng = StableRNG(123)
    W = rand(rng, Uniform(), n)
    T = rand(rng, [0, 1], n)
    Y = 3W .+ T .+ T.*W + rand(rng, Normal(0, 0.05), n)
    dataset =  DataFrame(
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

@testset "Test composition CM(1) - CM(0) = ATE(1,0)" begin
    dataset = make_dataset(;n=1000)
    CM₀ = CM(
        outcome = :Y,
        treatment_values = (T=0,),
        treatment_confounders = (T=[:W],)
    )
    CM₁ = CM(
        outcome = :Y,
        treatment_values = (T=1,),
        treatment_confounders = (T=[:W],)
    )
    mydiff(x) = x[2] - x[1]

    jointestimand = JointEstimand(CM₀, CM₁)
    models = Dict(
        :Y => with_encoder(LinearRegressor()),
        :T => LogisticClassifier(lambda=0)
    )
    tmle = Tmle(models=models)
    ose = Ose(models=models)
    cache = Dict()

    # Via Composition
    joint_tmle, cache = tmle(jointestimand, dataset; cache=cache, verbosity=0)
    diff_tmle = compose(mydiff, joint_tmle, AutoZygote())
    @test significance_test(diff_tmle) isa TMLE.OneSampleTTest
    joint_ose, cache = ose(jointestimand, dataset; cache=cache, verbosity=0)
    diff_ose = compose(mydiff, joint_ose, AutoZygote())
    @test significance_test(diff_ose) isa TMLE.OneSampleTTest

    # Via ATE
    ATE₁₀ = ATE(
        outcome = :Y,
        treatment_values = (T=(case=1, control=0),),
        treatment_confounders = (T=[:W],)
    )
    # Check composed TMLE
    ATE_tmle, cache = tmle(ATE₁₀, dataset; cache=cache, verbosity=0)
    @test estimate(ATE_tmle) ≈ first(estimate(diff_tmle)) atol = 1e-7
    # T Test
    diff_confint = collect(confint(OneSampleTTest(diff_tmle)))
    ATE_confint = collect(confint(OneSampleTTest(ATE_tmle)))
    @test ATE_confint ≈ diff_confint atol=1e-4
    # Z Test
    diff_confint = collect(confint(OneSampleZTest(diff_tmle)))
    ATE_confint = collect(confint(OneSampleZTest(ATE_tmle)))
    @test ATE_confint ≈ diff_confint atol=1e-4

    # Check composed OSE
    ATE_ose, cache = ose(ATE₁₀, dataset; cache=cache, verbosity=0)
    @test estimate(ATE_ose) ≈ only(estimate(diff_ose)) atol = 1e-7
    # T Test
    diff_confint = collect(confint(OneSampleTTest(diff_ose)))
    ATE_confint = collect(confint(OneSampleTTest(ATE_ose)))
    @test ATE_confint ≈ diff_confint atol=1e-4
    # Z Test
    diff_confint = collect(confint(OneSampleZTest(diff_ose)))
    ATE_confint = collect(confint(OneSampleZTest(ATE_ose)))
    @test ATE_confint ≈ diff_confint atol=1e-4
end

@testset "Test compose multidimensional function" begin
    dataset = make_dataset(;n=1000)
    models = Dict(
        :Y => with_encoder(LinearRegressor()),
        :T => LogisticClassifier(lambda=0)
    )
    tmle = Tmle(models=models)
    cache = Dict()
    
    joint = JointEstimand(
        CM(
        outcome = :Y,
        treatment_values = (T=1,),
        treatment_confounders = (T=[:W],)
        ),
        CM(
        outcome = :Y,
        treatment_values = (T=0,),
        treatment_confounders = (T=[:W],))
    )

    joint_estimate, cache = tmle(joint, dataset; cache=cache, verbosity=0)

    Main.eval(:(f(x) = [x[1]^2 - x[2], 2x[1] + 3x[2]]))

    composed_estimate = compose(Main.f, joint_estimate, AutoZygote())
    @test significance_test(composed_estimate) isa TMLE.OneSampleHotellingT2Test
    @test estimate(composed_estimate) == Main.f(estimate(joint_estimate))
    @test size(composed_estimate.cov) == (2, 2)

    composed_estimate_dict = TMLE.to_dict(composed_estimate)
    @test composed_estimate_dict isa Dict
    composed_estimate_from_dict = TMLE.from_dict!(composed_estimate_dict)
    @test composed_estimate_from_dict.estimand == composed_estimate.estimand
    @test composed_estimate_from_dict.estimates == composed_estimate.estimates
    @test composed_estimate_from_dict.cov == composed_estimate.cov
    @test composed_estimate_from_dict.n == composed_estimate.n
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
    dataset = DataFrame(
        W = W,
        T₁ = categorical(T₁),
        T₂ = categorical(T₂),
        Y = Y
    )
    AIE₁ = AIE(
        outcome = :Y,
        treatment_values = (T₁=(case=2, control=1), T₂=(case=2, control=1)),
        treatment_confounders = (T₁ = [:W], T₂ = [:W])
    )
    AIE₂ = AIE(
        outcome = :Y,
        treatment_values = (T₁=(case=3, control=1), T₂=(case=3, control=1)),
        treatment_confounders = (T₁ = [:W], T₂ = [:W])
    )
    AIE₃ = AIE(
        outcome = :Y,
        treatment_values = (T₁=(case=3, control=2), T₂=(case=3, control=2)),
        treatment_confounders = (T₁ = [:W], T₂ = [:W])
    )
    jointAIE = JointEstimand(AIE₁, AIE₂, AIE₃)

    ose = Ose(models=TMLE.default_models(G=LogisticClassifier(), Q_continuous=LinearRegressor()))
    jointEstimate, _ = ose(jointAIE, dataset, verbosity=0)

    testres = significance_test(jointEstimate)
    @test testres.x̄ ≈ estimate(jointEstimate)
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
    @test estimate(jointEstimate) == estimate(jointEstimate_fromdict)
    @test jointEstimate.n == jointEstimate_fromdict.n
    @test length(jointEstimate_fromdict.estimates) == 3

    if VERSION >= v"1.9"
        using JSON
        filename, _ = mktemp()
        TMLE.write_json(filename, jointEstimate)
        from_json = TMLE.read_json(filename, use_mmap=false)
        @test jointEstimate.estimand == from_json.estimand
        @test jointEstimate.cov == from_json.cov
        @test estimate(jointEstimate) == estimate(from_json)
        @test jointEstimate.n == from_json.n
        @test length(jointEstimate_fromdict.estimates) == 3
    end
end


end

true