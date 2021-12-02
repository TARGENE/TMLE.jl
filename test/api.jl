module TestAPI

using Test
using TMLE
using MLJ
using StableRNGs
using StatsBase
using Distributions


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
    F = binaryfluctuation(query=query)

    tmle = TMLEstimator(Q̅, G, F)

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
    F = binaryfluctuation(query=query)
    tmle = TMLEstimator(Q̅, G, F)

    mach = machine(tmle, T, W, y)
    fit!(mach, verbosity=0)

    # Test the various api results functions

    @test estimate(mach) ≈ -1.59 atol=1e-2
    @test stderror(mach) ≈ 1.32 atol=1e-2

    res = briefreport(mach)
    @test res.estimate ≈ -1.59 atol=1e-2
    @test res.stderror ≈ 1.32 atol=1e-2
    @test res.mean_inf_curve ≈ -1.52e-8 atol=1e-2
    @test res.pval ≈ 0.23 atol=1e-2
    confint = res.confint
    @test confint[1] ≈ -4.18 atol=1e-2
    @test confint[2] ≈ 1.01 atol=1e-2

    @test pvalue(mach) ≈ 0.23 atol=1e-2
    # Left is 1/2 as estimate < 0
    @test pvalue(mach, tail=:left) ≈ 0.115 atol=1e-3
    
    (lb, ub) = confinterval(mach)
    @test lb ≈ -4.18 atol=1e-2
    @test ub ≈ 1.01 atol=1e-2
end


@testset "Test Log p(T|W) is under threshold" begin
    n = 1000
    T = vcat(repeat([false], n), [true])
    T = (t₁=categorical(T),)
    W = MLJ.table(rand(n+1, 2))

    Gmach = machine(ConstantClassifier(), W, TMLE.adapt(T))
    fit!(Gmach, verbosity=0)

    query = (t₁=[true, false],)
    indicators = TMLE.indicator_fns(query)

    @test_logs (:info, "p(T|W) evaluated under 0.005 at indices: [1001]") TMLE.compute_covariate(Gmach, W, T, indicators; verbosity=1)

end

@testset "Test partial refit when changing query" begin
    n = 100
    p = 3
    W = MLJ.table(rand(n, p))
    T = (
        t₁=categorical(rand([true, false], n)),
        t₂=categorical(rand([true, false], n)),
    )
    y = categorical(rand([true, false], n))

    # First fit
    query = (t₁=[true, false], t₂=[true, false])
    Q̅ = ConstantClassifier()
    G = FullCategoricalJoint(ConstantClassifier())
    F = binaryfluctuation(query=query)
    tmle = TMLEstimator(Q̅, G, F)

    mach = machine(tmle, T, W, y)
    fit!(mach, verbosity=0)

    # Change the query, only Fluctuation and Report are refit
    tmle.F.query = (t₁=[false, true], t₂=[true, false])
    fit!(mach, verbosity=0)

    fp = fitted_params(mach)
    for m in fp.machines
        if m.model isa Union{Fluctuation, TMLE.Report}
            @test m.state == 2
        else
            @test m.state == 1
        end
    end

end


end;

true