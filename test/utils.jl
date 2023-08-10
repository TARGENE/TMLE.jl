module TestUtils

using Test
using Tables
using TableOperations
using TMLE
using MLJBase
using StableRNGs
using Distributions
using CategoricalArrays
using MLJGLMInterface: LinearBinaryClassifier
using MLJLinearModels
using MLJModels

@testset "Test estimand validation" begin
    dataset = (
        W₁ = [1., 2., 3.],
        T = categorical([1, 2, 3]), 
        Y = [1., 2., 3.]
        )
    # Treatment values absent from dataset
    scm = SCM(
        SE(:T, [:W₁]),
        SE(:Y, [:T, :W₁])
    )
    Ψ = ATE(
        scm,
        outcome=:Y,
        treatment=(T=(case=1, control=0),)
    )
    @test_throws TMLE.AbsentLevelError("T", "control", 0, [1, 2, 3]) TMLE.check_treatment_levels(Ψ, dataset)

    Ψ = CM(
        scm,
        outcome=:Y,
        treatment=(T=0,),
    )
    @test_throws TMLE.AbsentLevelError("T", 0, [1, 2, 3]) TMLE.check_treatment_levels(Ψ, dataset)
end

@testset "Test indicator_fns & indicator_values" begin
    scm = StaticConfoundedModel(
        [:Y],
        [:T₁, :T₂, :T₃],
        [:W]
    )
    dataset = (
        W  = [1, 2, 3, 4, 5, 6, 7, 8],
        T₁ = ["A", "B", "A", "B", "A", "B", "A", "B"],
        T₂ = [0, 0, 1, 1, 0, 0, 1, 1],
        T₃ = ["C", "C", "C", "C", "D", "D", "D", "D"],
        Y =  [1, 1, 1, 1, 1, 1, 1, 1]
    )
    # Conditional Mean
    Ψ = CM(
        scm,
        outcome=:Y, 
        treatment=(T₁="A", T₂=1),
    )
    indicator_fns = TMLE.indicator_fns(Ψ)
    @test indicator_fns == Dict(("A", 1) => 1.)
    indic_values = TMLE.indicator_values(indicator_fns, TMLE.treatments(dataset, Ψ))
    @test indic_values == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    # ATE
    Ψ = ATE(
        scm,
        outcome=:Y, 
        treatment=(T₁=(case="A", control="B"), T₂=(control=0, case=1)),
    )
    indicator_fns = TMLE.indicator_fns(Ψ)
    @test indicator_fns == Dict(
        ("A", 1) => 1.0,
        ("B", 0) => -1.0
    )
    indic_values = TMLE.indicator_values(indicator_fns, TMLE.treatments(dataset, Ψ))
    @test indic_values == [0.0, -1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0]
    # 2-points IATE
    Ψ = IATE(
        scm,
        outcome=:Y, 
        treatment=(T₁=(case="A", control="B"), T₂=(case=1, control=0)),
    )
    indicator_fns = TMLE.indicator_fns(Ψ)
    @test indicator_fns == Dict(
        ("A", 1) => 1.0,
        ("A", 0) => -1.0,
        ("B", 1) => -1.0,
        ("B", 0) => 1.0
    )
    indic_values = TMLE.indicator_values(indicator_fns, TMLE.treatments(dataset, Ψ))
    @test indic_values == [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0]
    # 3-points IATE
    Ψ = IATE(
        scm,
        outcome=:Y, 
        treatment=(T₁=(case="A", control="B"), T₂=(case=1, control=0), T₃=(control="D", case="C")),
    )
    indicator_fns = TMLE.indicator_fns(Ψ)
    @test indicator_fns == Dict(
        ("A", 1, "D") => -1.0,
        ("A", 1, "C") => 1.0,
        ("B", 0, "D") => -1.0,
        ("B", 0, "C") => 1.0,
        ("B", 1, "C") => -1.0,
        ("A", 0, "D") => 1.0,
        ("B", 1, "D") => 1.0,
        ("A", 0, "C") => -1.0
    )
    indic_values = TMLE.indicator_values(indicator_fns, TMLE.treatments(dataset, Ψ))
    @test indic_values == [-1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0]
end

@testset "Test expected_value" begin
    n = 100
    X = MLJBase.table(rand(n, 3))

    # Probabilistic Classifier
    y = categorical(rand([0, 1], n))
    mach = machine(ConstantClassifier(), X, y)
    fit!(mach; verbosity=0)
    proba = mach.fitresult[2][2]
    ŷ = MLJBase.predict(mach)
    expectation = TMLE.expected_value(ŷ)
    @test expectation == repeat([proba], n)

    # Probabilistic Regressor
    y = rand(n)
    mach = machine(ConstantRegressor(), X, y)
    fit!(mach; verbosity=0)
    ŷ = MLJBase.predict(mach)
    expectation = TMLE.expected_value(ŷ)
    @test expectation ≈ repeat([mean(y)], n) atol=1e-10

    # Deterministic Regressor
    mach = machine(LinearRegressor(), X, y)
    fit!(mach; verbosity=0)
    ŷ = MLJBase.predict(mach)
    expectation = TMLE.expected_value(ŷ)
    @test expectation == ŷ
end

@testset "Test counterfactualTreatment" begin
    vals = (true, "a")
    T = (
        T₁ = categorical([true, false, false], ordered=true),
        T₂ = categorical(["a", "a", "c"])
    )
    cfT = TMLE.counterfactualTreatment(vals, T)
    @test cfT == (
        T₁ = categorical([true, true, true]),
        T₂ = categorical(["a", "a", "a"])
    )
    @test isordered(cfT.T₁)
    @test !isordered(cfT.T₂)
end

@testset "Test Targeting sequence with 1 Treatment" begin
    ### In all cases, models are "constant" models
    ## First case: 1 Treamtent variable
    scm = StaticConfoundedModel(
       :Y, :T, [:W₁, :W₂, :W₃], 
       treatment_model = ConstantClassifier(),
       outcome_model = ConstantRegressor()
    )
    Ψ = ATE(
        scm,
        outcome=:Y, 
        treatment=(T=(case="a", control="b"),),
    )
    dataset = (
        T = categorical(["a", "b", "c", "a", "a", "b", "a"]),
        Y = [1., 2., 3, 4, 5, 6, 7],
        W₁ = rand(7),
        W₂ = rand(7),
        W₃ = rand(7)
    )
    fit!(Ψ, dataset, verbosity=0)
    indicator_fns = TMLE.indicator_fns(Ψ)
    T = TMLE.treatments(dataset, Ψ)
    @test T == (T=dataset.T,)
    W = TMLE.confounders(dataset, Ψ)
    @test W == (T = (W₁=dataset.W₁, W₂=dataset.W₂, W₃=dataset.W₃),)

    @test TMLE.compute_offset(Ψ) == repeat([mean(dataset.Y)], 7)
    weighted_fluctuation = true
    bw = TMLE.balancing_weights(scm, W, T)
    cov, w = TMLE.clever_covariate_and_weights(scm, W, T, indicator_fns, weighted_fluctuation=weighted_fluctuation)

    @test cov == [1.0, -1.0, 0.0, 1.0, 1.0, -1.0, 1.0]
    @test w == bw == [1.75, 3.5, 7.0, 1.75, 1.75, 3.5, 1.75]
    
    mach = TMLE.tmle_step(Ψ, weighted_fluctuation=weighted_fluctuation)
    @test fitted_params(mach) == (
        features = [:covariate],
        coef = [0.12500000000000003],
        intercept = 0.0
    )

    cov, w = TMLE.clever_covariate_and_weights(scm, W, T, indicator_fns)
    @test cov == [1.75, -3.5, 0.0, 1.75, 1.75, -3.5, 1.75]

    unweighted_mach = TMLE.tmle_step(Ψ, weighted_fluctuation=false)
    @test fitted_params(unweighted_mach) == (
        features = [:covariate],
        coef = [0.047619047619047616],
        intercept = 0.0
    )
end

@testset "Test Targeting sequence with 2 Treatments" begin
    scm = StaticConfoundedModel(
       [:Y], [:T₁, :T₂], [:W₁, :W₂, :W₃], 
       treatment_model = ConstantClassifier(),
       outcome_model = ConstantClassifier()
    )
    Ψ = IATE(
        scm,
        outcome =:Y, 
        treatment=(T₁=(case=1, control=0), T₂=(case=1, control=0)),
    )
    dataset = (
        T₁ = categorical([1, 0, 0, 1, 1, 1, 0]),
        T₂ = categorical([1, 1, 1, 1, 1, 0, 0]),
        Y = categorical([1, 1, 1, 1, 0, 0, 0]),
        W₁ = rand(7),
        W₂ = rand(7),
        W₃ = rand(7)
    )
    fit!(Ψ, dataset, verbosity=0)
    indicator_fns = TMLE.indicator_fns(Ψ)
    T = TMLE.treatments(dataset, Ψ)
    @test T == (T₁ = dataset.T₁, T₂ = dataset.T₂)
    W = TMLE.confounders(dataset, Ψ)
    @test W == (
        T₁ = (W₁ = dataset.W₁, W₂ = dataset.W₂, W₃ = dataset.W₃), 
        T₂ = (W₁ = dataset.W₁, W₂ = dataset.W₂, W₃ = dataset.W₃)
    )
    # Because the outcome is binary, the offset is the logit
    @test TMLE.compute_offset(Ψ) == repeat([0.28768207245178085], 7)
    
    cov, w = TMLE.clever_covariate_and_weights(scm, W, T, indicator_fns)
    @test cov ≈ [2.45, -3.266, -3.266, 2.45, 2.45, -6.125, 8.166] atol=1e-2
    @test w == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    mach = TMLE.tmle_step(Ψ)
    @test fitted_params(mach) == (
        features = [:covariate],
        coef = [-0.0945122832139119],
        intercept = 0.0
    )
end

@testset "Test Targeting sequence with 3 Treatments" begin
    ## Third case: 3 Treatment variables
    scm = StaticConfoundedModel(
       [:Y], [:T₁, :T₂, :T₃], [:W], 
       treatment_model = ConstantClassifier(),
       outcome_model = DeterministicConstantRegressor()
    )
    Ψ = IATE(
        scm,
        outcome =:Y, 
        treatment=(T₁=(case="a", control="b"), 
                   T₂=(case=1, control=2), 
                   T₃=(case=true, control=false)),
    )
    dataset = (
        T₁ = categorical(["a", "a", "b", "b", "c", "b", "b"]),
        T₂ = categorical([3, 2, 1, 1, 2, 2, 2], ordered=true),
        T₃ = categorical([true, false, true, false, false, false, false], ordered=true),
        Y  = [1., 2., 3, 4, 5, 6, 7],
        W  = rand(7),
    )

    fit!(Ψ, dataset, verbosity=0)
    indicator_fns = TMLE.indicator_fns(Ψ)
    T = TMLE.treatments(dataset, Ψ)
    @test T == (T₁ = dataset.T₁, T₂ = dataset.T₂, T₃ = dataset.T₃)
    W = TMLE.confounders(dataset, Ψ)
    @test W == (
        T₁ = (W = dataset.W,), 
        T₂ = (W = dataset.W,),
        T₃ = (W = dataset.W,)
    )
    indicator_fns = TMLE.indicator_fns(Ψ)

    @test TMLE.compute_offset(Ψ) == repeat([4.0], 7)

    cov, w = TMLE.clever_covariate_and_weights(scm, W, T, indicator_fns)
    @test cov ≈ [0, 8.575, -21.4375, 8.575, 0, -4.2875, -4.2875] atol=1e-3
    @test w == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    mach = TMLE.tmle_step(Ψ)
    @test fitted_params(mach) == (
        features = [:covariate],
        coef = [-0.02665556018325698],
        intercept = 0.0
    )
end

@testset "Test fluctuation_input" begin
    X = TMLE.fluctuation_input([1., 2.], [1., 2])
    @test X.covariate isa Vector{Float64}
    @test X.offset isa Vector{Float64}

    X = TMLE.fluctuation_input([1., 2.], [1.f0, 2.f0])
    @test X.covariate isa Vector{Float64}
    @test X.offset isa Vector{Float64}

    X = TMLE.fluctuation_input([1.f0, 2.f0], [1., 2.])
    @test X.covariate isa Vector{Float32}
    @test X.offset isa Vector{Float32}

end

end;

true