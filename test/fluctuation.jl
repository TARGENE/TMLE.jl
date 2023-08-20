module TestFluctuation

using Test
using TMLE
using MLJModels
using MLJBase

function test_fluctuation_fit(Ψ, ps_lowerbound, weighted_fluctuation)
    Q⁰ = TMLE.getQ(Ψ)
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
    @test Q.fitresult.data[1] == Xfluct
    @test Q.fitresult.data[3] == weights
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
    fit!(Ψ, dataset, verbosity=0)

    ps_lowerbound = 1e-8
    weighted_fluctuation = true
    test_fluctuation_fit(Ψ, ps_lowerbound, weighted_fluctuation)

    weighted_fluctuation = false
    test_fluctuation_fit(Ψ, ps_lowerbound, weighted_fluctuation)
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

    fit!(Ψ, dataset, verbosity=0)
    test_fluctuation_fit(Ψ, ps_lowerbound, weighted_fluctuation)
end

@testset "Test Fluctuation with 3 Treatments" begin
    ## Third case: 3 Treatment variables
    Ψ = IATE(
        outcome =:Y, 
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

    fit!(Ψ, dataset, verbosity=0)
    test_fluctuation_fit(Ψ, ps_lowerbound, weighted_fluctuation)
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