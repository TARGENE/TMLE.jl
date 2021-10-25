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

    tmle = TMLEstimator(Q̅, G, :binary, query)

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
    tmle = TMLEstimator(Q̅, G, :binary, query)

    mach = machine(tmle, T, W, y)
    fit!(mach, verbosity=0)

    # Test the various api results functions

    @test estimate(mach) ≈ -1.59 atol=1e-2
    @test stderror(mach) ≈ 1.32 atol=1e-2

    res = briefreport(mach)
    @test res.estimate ≈ -1.59 atol=1e-2
    @test res.stderror ≈ 1.32 atol=1e-2
    @test res.mean_inf_curve ≈ -1.52e-8 atol=1e-2

    @test pvalue(mach) ≈ 0.23 atol=1e-2
    
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
    tmle = TMLEstimator(ConstantClassifier(),
                        ConstantClassifier(),
                        :binary, 
                        query)
    @test_logs (:info, "p(T|W) evaluated under 0.005 at indices: [1001]") TMLE.compute_covariate(tmle, Gmach, W, T; verbosity=1)

end


@testset "Test setproperty!(query) sets indicator functions" begin
    query = (t₁=[true, false],)
    tmle = TMLEstimator(ConstantClassifier(),
            ConstantClassifier(),
            :binary, 
            query)

    @test tmle.indicators == Dict(
        (t₁ = 1,) => 1,
        (t₁ = 0,) => -1
    )

    tmle.query = (t₁=[false, true], t₂=["a", "b"])

    @test tmle.indicators == Dict(
        (t₁ = 1, t₂ = "a") => -1,
        (t₁ = 0, t₂ = "b") => -1,
        (t₁ = 0, t₂ = "a") => 1,
        (t₁ = 1, t₂ = "b") => 1
    )

    @test_throws ArgumentError tmle.indicators = Dict()
end

end;

true