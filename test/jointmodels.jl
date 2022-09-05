module TestJointModels

using Test
using TMLE
using StatsBase
using StableRNGs
using MLJBase
using MLJLinearModels

@testset "Test FullCategoricalJoint" begin
    rng = StableRNG(123)
    n = 10
    X = rand(n, 4)
    Y = categorical(sample(rng, ["A", "G", "C"], (n, 2)))
    Y = (Y₁ = Y[:, 1], Y₂ = Y[:, 2])
    
    jointmodel = TMLE.FullCategoricalJoint(LogisticClassifier(lambda=0))
    mach = machine(jointmodel, MLJBase.table(X), Y)
    fit!(mach, verbosity=0)

    # The encoding should reflect all combinations
    @test mach.fitresult.encoding == Dict(
        ("C", "C") => 5,
        ("C", "A") => 2,
        ("A", "C") => 4,
        ("A", "G") => 7,
        ("G", "C") => 6,
        ("C", "G") => 8,
        ("G", "A") => 3,
        ("G", "G") => 9,
        ("A", "A") => 1
        )
    # The underlying model should have been fitted
    @test mach.fitresult.model_fitresult[2] == (:x1, :x2, :x3, :x4)

    # Only a few of the categories are actually present in the data
    y_multi = TMLE.encode(Y, mach)
    @test y_multi == categorical([5, 3, 6, 4, 9, 9, 6, 9, 3, 7])

    ypred = MLJBase.predict(mach)
    @test ypred[1] isa MLJBase.UnivariateFinite

    d = TMLE.density(mach, X, Y)
    @test d == [pdf(p, y_multi[i]) for (i, p) in enumerate(ypred)]
end

@testset "Test density fallback" begin
    rng = StableRNG(123)
    n = 10
    X = rand(rng, n, 4)
    y = categorical(sample(rng, ["A", "G", "C"], n))
    mach = machine(LogisticClassifier(lambda=1), MLJBase.table(X), y)
    fit!(mach, verbosity=0)

    d = TMLE.density(mach, X, y)

    @test d ≈ [0.565,
                0.218,
                0.256,
                0.573,
                0.286,
                0.612,
                0.599,
                0.604,
                0.295,
                0.656] atol=1e-2
end

end

true