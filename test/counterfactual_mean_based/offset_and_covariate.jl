module TestOffsetAndCovariate

using Test
using TMLE
using CategoricalArrays
using MLJModels
using Statistics

@testset "Test ps_lower_bound" begin
    dataset = (
        T = categorical([1, 0, 1, 1, 0, 1, 1]),
        Y = [1., 2., 3, 4, 5, 6, 7],
        W = rand(7),
    )
    Ψ = ATE(
        outcome=:Y, 
        treatment=(T=(case=1, control=0),),
        confounders=:W,
    )
    max_lb = 0.1
    fit!(Ψ, dataset, verbosity=0)
    # Nothing results in data adaptive lower bound no lower than max_lb
    @test TMLE.ps_lower_bound(Ψ, nothing) == max_lb
    @test TMLE.data_adaptive_ps_lower_bound(Ψ) == max_lb
    @test TMLE.data_adaptive_ps_lower_bound(1000) == 0.02984228238321508
    # Otherwise use the provided threhsold provided it's lower than max_lb
    @test TMLE.ps_lower_bound(Ψ, 1e-8) == 1.0e-8
    @test TMLE.ps_lower_bound(Ψ, 1) == 0.1
end

@testset "Test compute_offset, clever_covariate_and_weights: 1 treatment" begin
    ### In all cases, models are "constant" models
    ## First case: 1 Treamtent variable
    Ψ = ATE(
        outcome=:Y, 
        treatment=(T=(case="a", control="b"),),
        confounders=[:W₁, :W₂, :W₃],
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

    offset = TMLE.compute_offset(Ψ)
    @test offset == repeat([mean(dataset.Y)], 7)
    weighted_fluctuation = true
    ps_lowerbound = 1e-8
    X, y = TMLE.get_outcome_datas(Ψ)
    cov, w = TMLE.clever_covariate_and_weights(Ψ, X;
        ps_lowerbound=ps_lowerbound,
        weighted_fluctuation=weighted_fluctuation
    )

    @test cov == [1.0, -1.0, 0.0, 1.0, 1.0, -1.0, 1.0]
    @test w == [1.75, 3.5, 7.0, 1.75, 1.75, 3.5, 1.75]

    weighted_fluctuation = false
    cov, w = TMLE.clever_covariate_and_weights(Ψ, X;
        ps_lowerbound=ps_lowerbound,
        weighted_fluctuation=weighted_fluctuation
    )
    @test cov == [1.75, -3.5, 0.0, 1.75, 1.75, -3.5, 1.75]
    @test w == ones(7)
end

@testset "Test compute_offset, clever_covariate_and_weights: 2 treatments" begin
    Ψ = IATE(
        outcome = :Y,
        treatment=(T₁=(case=1, control=0), T₂=(case=1, control=0)),
        confounders=[:W₁, :W₂, :W₃],
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

    # Because the outcome is binary, the offset is the logit
    offset = TMLE.compute_offset(Ψ)
    @test offset == repeat([0.28768207245178085], 7)
    X, y = TMLE.get_outcome_datas(Ψ)
    cov, w = TMLE.clever_covariate_and_weights(Ψ, X;
        ps_lowerbound=ps_lowerbound,
        weighted_fluctuation=weighted_fluctuation
    )
    @test cov ≈ [2.45, -3.266, -3.266, 2.45, 2.45, -6.125, 8.166] atol=1e-2
    @test w == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
end

@testset "Test compute_offset, clever_covariate_and_weights: 3 treatments" begin
    ## Third case: 3 Treatment variables
    Ψ = IATE(
        outcome =:Y, 
        treatment=(T₁=(case="a", control="b"), 
                   T₂=(case=1, control=2), 
                   T₃=(case=true, control=false)),
        confounders=[:W],
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

    offset = TMLE.compute_offset(Ψ)
    @test offset == repeat([4.0], 7)
    X, y = TMLE.get_outcome_datas(Ψ)
    cov, w = TMLE.clever_covariate_and_weights(Ψ, X;
        ps_lowerbound=ps_lowerbound, 
        weighted_fluctuation=weighted_fluctuation
    )
    @test cov ≈ [0, 8.575, -21.4375, 8.575, 0, -4.2875, -4.2875] atol=1e-3
    @test w == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
end

end

true