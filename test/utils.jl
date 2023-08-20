module TestUtils

using Test
using TMLE
using MLJBase
using StableRNGs
using CategoricalArrays
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

end;

true