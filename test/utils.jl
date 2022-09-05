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

@testset "Test indicator_fns" begin
    # Conditional Mean
    Ψ = CM(
        target=:y, 
        treatment=(T₁="A", T₂=1),
        confounders=[:W]
    )
    @test TMLE.indicator_fns(Ψ) == Dict(("A", 1) => 1)
    # ATE
    Ψ = ATE(
        target=:y, 
        treatment=(T₁=(case="A", control="B"), T₂=(control=0, case=1)),
        confounders=[:W]
    )
    @test TMLE.indicator_fns(Ψ) == Dict(
        ("A", 1) => 1,
        ("B", 0) => -1
    )
    # 2-points IATE
    Ψ = IATE(
        target=:y, 
        treatment=(T₁=(case="A", control="B"), T₂=(case=1, control=0)),
        confounders=[:W]
    )
    @test TMLE.indicator_fns(Ψ) == Dict(
        ("A", 1) => 1,
        ("A", 0) => -1,
        ("B", 1) => -1,
        ("B", 0) => 1
    )
    # 3-points IATE
    Ψ = IATE(
        target=:y, 
        treatment=(T₁=(case="A", control="B"), T₂=(case=1, control=0), T₃=(control="D", case="C")),
        confounders=[:W]
    )
    @test TMLE.indicator_fns(Ψ) == Dict(
        ("A", 1, "D") => -1,
        ("A", 1, "C") => 1,
        ("B", 0, "D") => -1,
        ("B", 0, "C") => 1,
        ("B", 1, "C") => -1,
        ("A", 0, "D") => 1,
        ("B", 1, "D") => 1,
        ("A", 0, "C") => -1
    )
end


@testset "Test expected_value & maybelogit" begin
    n = 100
    X = MLJBase.table(rand(n, 3))

    # Probabilistic Classifier
    y = categorical(rand([0, 1], n))
    mach = machine(ConstantClassifier(), X, y)
    fit!(mach; verbosity=0)
    proba = mach.fitresult[2][2]
    ŷ = MLJBase.predict(mach)
    expectation = TMLE.expected_value(ŷ, typeof(mach.model), target_scitype(mach.model))
    @test expectation == repeat([proba], n)
    @test TMLE.maybelogit(expectation, typeof(mach.model), target_scitype(mach.model)) == TMLE.logit(expectation)

    # Probabilistic Regressor
    y = rand(n)
    mach = machine(ConstantRegressor(), X, y)
    fit!(mach; verbosity=0)
    ŷ = MLJBase.predict(mach)
    expectation = TMLE.expected_value(ŷ, typeof(mach.model), target_scitype(mach.model))
    @test expectation ≈ repeat([mean(y)], n) atol=1e-10
    @test TMLE.maybelogit(expectation, typeof(mach.model), target_scitype(mach.model)) == expectation

    # Deterministic Regressor
    mach = machine(LinearRegressor(), X, y)
    fit!(mach; verbosity=0)
    ŷ = MLJBase.predict(mach)
    expectation = TMLE.expected_value(ŷ, typeof(mach.model), target_scitype(mach.model))
    @test expectation == ŷ
    @test TMLE.maybelogit(expectation, typeof(mach.model), target_scitype(mach.model)) == expectation
end

@testset "Test adapt" begin
    T = (a=1,)
    @test TMLE.adapt(T) == 1

    T = (a=1, b=2)
    @test TMLE.adapt(T) == T
end

@testset "Test indicator_values" begin
    indicators = Dict(
        ("b", "c", 1) => -1,
        ("a", "c", 1) => 1,
        ("b", "d", 0) => -1,
        ("b", "c", 0) => 1,
        ("a", "d", 1) => -1,
        ("a", "c", 0) => -1,
        ("a", "d", 0) => 1,
        ("b", "d", 1) => 1 
    )
    T = (
        t₁= categorical(["b", "a", "b", "b", "a", "a", "a", "b", "q"]),
        t₂ = categorical(["c", "c", "d", "c", "d", "c", "d", "d", "d"]),
        t₃ = categorical([true, true, false, false, true, false, false, true, false])
        )
    # The las combination does not appear in the indicators
    @test TMLE.indicator_values(indicators, T) ==
        [-1, 1, -1, 1, -1, -1, 1, 1, 0]
    # @btime TMLE.indicator_values(indicators, T)
    # @btime TMLE._indicator_values(indicators, T)
end


@testset "Test counterfactualTreatment" begin
    vals = (true, "a")
    T = (
        t₁ = categorical([true, false, false]),
        t₂ = categorical(["a", "a", "c"])
    )
    cfT = TMLE.counterfactualTreatment(vals, T)
    @test cfT == (
        t₁ = categorical([true, true, true]),
        t₂ = categorical(["a", "a", "a"])
    )
end

@testset "Test compute_covariate" begin
    # First case: 1 categorical variable
    # Using a trivial classifier
    # that outputs the proportions of of the classes
    T = (t₁ = categorical(["a", "b", "c", "a", "a", "b", "a"]),)
    W = MLJBase.table(rand(7, 3))

    Gmach = machine(ConstantClassifier(), 
                    W,
                    TMLE.adapt(T))
    fit!(Gmach, verbosity=0)

    Ψ = ATE(
        target =:y, 
        treatment=(t₁=(case="a", control="b"),),
        confounders = [:x1, :x2, :x3]
    )

    indicators = TMLE.indicator_fns(Ψ)

    cov = TMLE.compute_covariate(Gmach, W, T, indicators)
    @test cov == [1.75,
                 -3.5,
                 0.0,
                 1.75,
                 1.75,
                 -3.5,
                 1.75]

    # Second case: 2 binary variables
    # Using a trivial classifier
    # that outputs the proportions of of the classes
    T = (t₁ = categorical([1, 0, 0, 1, 1, 1, 0]),
         t₂ = categorical([1, 1, 1, 1, 1, 0, 0]))
    W = MLJBase.table(rand(7, 3))

    Gmach = machine(TMLE.FullCategoricalJoint(ConstantClassifier()), 
                    W, 
                    T)
    fit!(Gmach, verbosity=0)
    Ψ = IATE(
        target =:y, 
        treatment=(t₁=(case=1, control=0), t₂=(case=1, control=0)),
        confounders = [:x1, :x2, :x3]
    )
    indicators = TMLE.indicator_fns(Ψ)

    cov = TMLE.compute_covariate(Gmach, W, T, indicators)
    @test cov == [2.3333333333333335,
                 -3.5,
                 -3.5,
                 2.3333333333333335,
                 2.3333333333333335,
                 -7.0,
                 7.0]

    # Third case: 3 mixed categorical variables
    # Using a trivial classifier
    # that outputs the proportions of of the classes
    T = (t₁ = categorical(["a", "a", "b", "b", "c", "b", "b"]),
         t₂ = categorical([3, 2, 1, 1, 2, 2, 2]),
         t₃ = categorical([true, false, true, false, false, false, false]))
    W = MLJBase.table(rand(7, 3))

    Gmach = machine(TMLE.FullCategoricalJoint(ConstantClassifier()), 
                    W, 
                    T)
    fit!(Gmach, verbosity=0)
    Ψ = IATE(
        target =:y, 
        treatment=(t₁=(case="a", control="b"), 
                   t₂=(case=1, control=2), 
                   t₃=(case=true, control=false)),
        confounders = [:x1, :x2, :x3]
    )

    indicators = TMLE.indicator_fns(Ψ)

    cov = TMLE.compute_covariate(Gmach, W, T, indicators)
    @test cov == [0,
                  7.0,
                 -7,
                  7,
                  0,
                 -3.5,
                 -3.5]
end

@testset "Test compute_offset" begin
    n = 10
    X = rand(n, 3)

    # When Y is binary
    y = categorical([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    mach = machine(ConstantClassifier(), MLJBase.table(X), y)
    fit!(mach, verbosity=0)
    # Should be equal to logit(Ê[Y|X])= logit(4/10) = -0.4054651081081643
    @test TMLE.compute_offset(mach, X) == repeat([-0.4054651081081643], n)

    # When Y is continuous
    y = [1., 2., 3, 4, 5, 6, 7, 8, 9, 10]
    mach = machine(MLJModels.DeterministicConstantRegressor(), MLJBase.table(X), y)
    fit!(mach, verbosity=0)
    # Should be equal to Ê[Y|X] = 5.5
    @test TMLE.compute_offset(mach, X) == repeat([5.5], n)
    
end

@testset "Test logit" begin
    @test TMLE.logit([0.4, 0.8, 0.2]) ≈ [
        -0.40546510810,
        1.38629436112,
        -1.38629436112
    ]
    @test TMLE.logit([1, 0]) == [Inf, -Inf]
end

end;

true