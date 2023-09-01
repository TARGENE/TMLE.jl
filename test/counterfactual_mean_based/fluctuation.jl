module TestFluctuation

using Test
using TMLE
using MLJModels
using MLJBase

function test_fluctuation_fit(Ψ, ps_lowerbound, weighted_fluctuation)
    Q⁰ = TMLE.get_outcome_model(Ψ)
    X, y = Q⁰.data
    Q = machine(
        TMLE.Fluctuation(Ψ, 0.1, ps_lowerbound, weighted_fluctuation), 
        X, 
        y
    )
    fit!(Q, verbosity=0)
    Xfluct, weights = TMLE.clever_covariate_offset_and_weights(Ψ, Q⁰, X; 
        ps_lowerbound=ps_lowerbound, 
        weighted_fluctuation=weighted_fluctuation
    )
    @test Q.cache.weighted_covariate == Xfluct.covariate .* weights
end


@testset "Test Fluctuation with 1 Treatments" begin
    ### In all cases, models are "constant" models
    ## First case: 1 Treamtent variable
    Ψ = ATE(
        outcome=:Y,
        confounders=[:W₁, :W₂, :W₃],
        treatment=(T=(case="a", control="b"),),
        treatment_model = ConstantClassifier(),
        outcome_model = ConstantRegressor()
    )
    dataset = (
        T = categorical(["a", "b", "c", "a", "a", "b", "a"]),
        Y = [1., 2., 3, 4, 5, 6, 7],
        W₁ = rand(7),
        W₂ = rand(7),
        W₃ = rand(7)
    )
    initial_factors = fit!(Ψ, dataset, verbosity=0)
    ps_lowerbound = 1e-8
    weighted_fluctuation = true
    targeted_factors = TMLE.fluctuate(initial_factors, Ψ; 
        ps_lowerbound=ps_lowerbound,
        weighted_fluctuation=weighted_fluctuation,
        verbosity=1,
        tol=nothing
    )
    @test targeted_factors.T === initial_factors.T
    @test targeted_factors.Y !== initial_factors.Y
    @test targeted_factors.Y.machine.cache.weighted_covariate ==  [1.75, -3.5, 0.0, 1.75, 1.75, -3.5, 1.75]
    Q = targeted_factors.Y
    X = Q.machine.data[1]
    Xfluct, weights = TMLE.clever_covariate_offset_and_weights(Q.model, X)
    @test weights == [1.75, 3.5, 7.0, 1.75, 1.75, 3.5, 1.75]
    @test targeted_factors.Y.machine.cache.weighted_covariate == Xfluct.covariate .* weights

    weighted_fluctuation = false
    targeted_factors = TMLE.fluctuate(initial_factors, Ψ; 
        ps_lowerbound=ps_lowerbound,
        weighted_fluctuation=weighted_fluctuation,
        verbosity=1,
        tol=nothing
    )
    Q = targeted_factors.Y
    X = Q.machine.data[1]
    Xfluct, weights = TMLE.clever_covariate_offset_and_weights(Q.model, X)
    @test weights == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
end

@testset "Test Fluctuation with 2 Treatments" begin
    Ψ = IATE(
        outcome =:Y, 
        treatment=(T₁=(case=1, control=0), T₂=(case=1, control=0)),
        confounders = [:W₁, :W₂, :W₃],
        treatment_model = ConstantClassifier(),
        outcome_model = ConstantClassifier()
    )
    dataset = (
        T₁ = categorical([1, 0, 0, 1, 1, 1, 0]),
        T₂ = categorical([1, 1, 1, 1, 1, 0, 0]),
        Y = categorical([1, 1, 1, 1, 0, 0, 0]),
        W₁ = rand(7),
        W₂ = rand(7),
        W₃ = rand(7)
    )
    ps_lowerbound = 1e-8
    weighted_fluctuation = false

    initial_factors = fit!(Ψ, dataset, verbosity=0)
    targeted_factors = TMLE.fluctuate(initial_factors, Ψ; 
        ps_lowerbound=ps_lowerbound,
        weighted_fluctuation=weighted_fluctuation,
        verbosity=1,
        tol=nothing
    )
    @test targeted_factors.T₁ === initial_factors.T₁
    @test targeted_factors.T₂ === initial_factors.T₂
    @test targeted_factors.Y !== initial_factors.Y
    @test targeted_factors.Y.machine.cache.weighted_covariate ≈ 
        [2.45, -3.27, -3.27, 2.45, 2.45, -6.13, 8.17] atol=0.01
end

@testset "Test Fluctuation with 3 Treatments" begin
    ## Third case: 3 Treatment variables
    Ψ = IATE(
        outcome = :Y, 
        treatment=(T₁=(case="a", control="b"), 
                   T₂=(case=1, control=2), 
                   T₃=(case=true, control=false)),
        confounders=:W,
        treatment_model = ConstantClassifier(),
        outcome_model = DeterministicConstantRegressor()
    )
    dataset = (
        T₁ = categorical(["a", "a", "b", "b", "c", "b", "b"]),
        T₂ = categorical([3, 2, 1, 1, 2, 2, 2], ordered=true),
        T₃ = categorical([true, false, true, false, false, false, false], ordered=true),
        Y  = [1., 2., 3, 4, 5, 6, 7],
        W  = rand(7),
    )
    ps_lowerbound = 1e-8 
    weighted_fluctuation = false

    initial_factors = fit!(Ψ, dataset, verbosity=0)
    targeted_factors = TMLE.fluctuate(initial_factors, Ψ; 
        ps_lowerbound=ps_lowerbound,
        weighted_fluctuation=weighted_fluctuation,
        verbosity=1,
        tol=nothing
    )
    @test targeted_factors.T₁ === initial_factors.T₁
    @test targeted_factors.T₂ === initial_factors.T₂
    @test targeted_factors.T₃ === initial_factors.T₃
    @test targeted_factors.Y !== initial_factors.Y
    @test targeted_factors.Y.machine.cache.weighted_covariate ≈
        [0.0, 8.58, -21.44, 8.58, 0.0, -4.29, -4.29] atol=0.01
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

end

true