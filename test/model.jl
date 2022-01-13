module TestModel

using Test
using TMLE
using MLJ
using StableRNGs
using StatsBase
using Distributions

LinearBinaryClassifier = @load LinearBinaryClassifier pkg=GLM verbosity=0

@testset "Test sub machines have been fitted correctly" begin
    rng = StableRNG(123)
    n = 100
    T = (t₁=categorical(sample(rng, ["CG", "CC"], Weights([0.7, 0.3]), n)),
         t₂=categorical(sample(rng, ["AT", "AA"], Weights([0.6, 0.4]), n)))
    W = (w₁=rand(rng, n), w₂=rand(rng, n))
    y = categorical(rand(rng, Bernoulli(0.3), n))

    query = (t₁=["CC", "CG"], t₂=["AT", "AA"])
    Q̅ = ConstantClassifier()
    G = FullCategoricalJoint(ConstantClassifier())
    F = LinearBinaryClassifier(fit_intercept=false, offsetcol=:offset)

    tmle = TMLEstimator(Q̅, G, query)

    mach = machine(tmle, T, W, y)
    fit!(mach, verbosity=0)

    # Fit outside of tmle
    Hmach = machine(OneHotEncoder(drop_last=true), T)
    fit!(Hmach, verbosity=0)
    Thot = transform(Hmach, T)
    X = merge(Thot, W)
    Q̅mach = machine(Q̅, X, y)
    fit!(Q̅mach, verbosity=0)
    @test Q̅mach.fitresult == fitted_params(mach).Q̅.target_distribution

    Gmach = machine(G, W, T)
    fit!(Gmach, verbosity=0)
    @test Gmach.fitresult == fitted_params(mach).G.fitresult
end


@testset "Test 4-points estimation non regression" begin
    rng = StableRNG(123)
    n = 100
    T = (t₁=categorical(sample(rng, ["CG", "CC"], Weights([0.7, 0.3]), n)),
         t₂=categorical(sample(rng, ["AT", "AA", "TT"], Weights([0.5, 0.4, 0.1]), n)),
         t₃=categorical(sample(rng, ["CC", "GG", "CG"], Weights([0.6, 0.2, 0.2]), n)),
         t₄=categorical(sample(rng, ["TT", "AA"], Weights([0.6, 0.4]), n))
         )
    W = (w₁=rand(rng, n), w₂=rand(rng, n))
    y = categorical(rand(rng, Bernoulli(0.3), n))

    query = (t₁=["CC", "CG"], t₂=["AT", "AA"], t₃=["CC", "GG"], t₄=["TT", "AA"])
    Q̅ = ConstantClassifier()
    G = FullCategoricalJoint(ConstantClassifier())
    tmle = TMLEstimator(Q̅, G, query)

    mach = machine(tmle, T, W, y)
    fit!(mach, verbosity=0)

    # Test the various api results functions

    res = briefreport(mach)[1]
    @test res.estimate ≈ -1.59 atol=1e-2
    @test res.stderror ≈ 1.32 atol=1e-2
    @test res.mean_inf_curve ≈ -1.52e-8 atol=1e-2
    @test res.pvalue ≈ 0.23 atol=1e-2
    @test res.confint[1] ≈ -4.18 atol=1e-2
    @test res.confint[2] ≈ 1.01 atol=1e-2
end


end;

true