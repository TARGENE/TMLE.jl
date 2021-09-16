module TestJointModels

using Test
using TMLE
using MLJ
using StatsBase
using StableRNGs
using CategoricalArrays


LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels verbosity=0


@testset "Test FullCategoricalJoint" begin
    rng = StableRNG(123)
    n = 10
    X = rand(n, 4)
    Y = categorical(sample(rng, ["A", "G", "C"], (n, 2)))
    Y = (Y₁ = Y[:, 1], Y₂ = Y[:, 2])
    
    jointmodel = FullCategoricalJoint(LogisticClassifier())
    mach = machine(jointmodel, MLJ.table(X), Y)
    fit!(mach)

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
    y_multi = TMLE.encode(Y, mach.fitresult.encoding)
    @test y_multi == categorical([5, 3, 6, 4, 9, 9, 6, 9, 3, 7])

    ypred = MLJ.predict(mach)
    @test ypred isa MLJ.UnivariateFiniteVector

    d = TMLE.density(mach, X, Y)
    @test d == [pdf(p, y_multi[i]) for (i, p) in enumerate(ypred)]

    println("WARNING: Need to add a further test as pdf will not work if the query is itself categorical")

end
end

true