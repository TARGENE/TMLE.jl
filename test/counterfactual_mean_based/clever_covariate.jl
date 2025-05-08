module TestOffsetAndCovariate

using Test
using TMLE
using CategoricalArrays
using MLJModels
using DataFrames

@testset "Test ps_lower_bound" begin
    n = 7
    max_lb = 0.1
    # Nothing results in data adaptive lower bound no lower than max_lb
    @test TMLE.ps_lower_bound(n, nothing) == max_lb
    @test TMLE.data_adaptive_ps_lower_bound(n) == max_lb
    @test TMLE.data_adaptive_ps_lower_bound(1000) == 0.02984228238321508
    # Otherwise use the provided threhsold provided it's lower than max_lb
    @test TMLE.ps_lower_bound(n, 1e-8) == 1.0e-8
    @test TMLE.ps_lower_bound(n, 1) == 0.1
end

@testset "Test clever_covariate_and_weights: 1 treatment" begin
    Ψ = ATE(
        outcome=:Y, 
        treatment_values=(T=(case="a", control="b"),),
        treatment_confounders=(T=[:W],),
    )
    dataset = DataFrame(
        T = categorical(["a", "b", "c", "a", "a", "b", "a"]),
        Y = [1., 2., 3, 4, 5, 6, 7],
        W = rand(7),
    )

    propensity_score_estimator = TMLE.JointConditionalDistributionEstimator(Dict(:T => TMLE.MLConditionalDistributionEstimator(ConstantClassifier())))
    propensity_score_estimate = propensity_score_estimator(
        (TMLE.ConditionalDistribution(:T, [:W]),),
        dataset,
        verbosity=0
    )
    weighted_fluctuation = true
    ps_lowerbound = 1e-8
    cov, w = TMLE.clever_covariate_and_weights(Ψ, propensity_score_estimate, dataset; 
        ps_lowerbound=ps_lowerbound, 
        weighted_fluctuation=weighted_fluctuation
    )

    @test cov == [1.0, -1.0, 0.0, 1.0, 1.0, -1.0, 1.0]
    @test w == [1.75, 3.5, 7.0, 1.75, 1.75, 3.5, 1.75]

    weighted_fluctuation = false
    cov, w = TMLE.clever_covariate_and_weights(Ψ, propensity_score_estimate, dataset;
        ps_lowerbound=ps_lowerbound,
        weighted_fluctuation=weighted_fluctuation
    )
    @test cov == [1.75, -3.5, 0.0, 1.75, 1.75, -3.5, 1.75]
    @test w == ones(7)
end

@testset "Test clever_covariate_and_weights: 2 treatments" begin
    Ψ = AIE(
        outcome = :Y,
        treatment_values=(
            T₁=(case=1, control=0), 
            T₂=(case=1, control=0)
        ),
        treatment_confounders=(T₁=[:W], T₂=[:W])
    )
    dataset = DataFrame(
        T₁ = categorical([1, 0, 0, 1, 1, 1, 0]),
        T₂ = categorical([1, 1, 1, 1, 1, 0, 0]),
        Y = categorical([1, 1, 1, 1, 0, 0, 0]),
        W = rand(7)
    )
    distribution_estimator = TMLE.MLConditionalDistributionEstimator(ConstantClassifier())
    propensity_score_estimator = TMLE.JointConditionalDistributionEstimator(Dict(
        :T₁ => distribution_estimator,
        :T₂ => distribution_estimator
    ))
    propensity_score_estimate = propensity_score_estimator(
        (TMLE.ConditionalDistribution(:T₁, [:W]), TMLE.ConditionalDistribution(:T₂, [:W])),
        dataset,
        verbosity=0
    )

    ps_lowerbound = 1e-8
    weighted_fluctuation = false

    cov, w = TMLE.clever_covariate_and_weights(Ψ, propensity_score_estimate, dataset;
        ps_lowerbound=ps_lowerbound,
        weighted_fluctuation=weighted_fluctuation
    )
    @test cov ≈ [2.45, -3.266, -3.266, 2.45, 2.45, -6.125, 8.166] atol=1e-2
    @test w == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
end

@testset "Test compute_offset, clever_covariate_and_weights: 3 treatments" begin
    ## Third case: 3 Treatment variables
    Ψ = AIE(
        outcome =:Y, 
        treatment_values=(
            T₁=(case="a", control="b"), 
            T₂=(case=1, control=2), 
            T₃=(case=true, control=false)
        ),
        treatment_confounders=(
            T₁=[:W],
            T₂=[:W],
            T₃=[:W]
        )
    )
    dataset = DataFrame(
        T₁ = categorical(["a", "a", "b", "b", "c", "b", "b"]),
        T₂ = categorical([3, 2, 1, 1, 2, 2, 2], ordered=true),
        T₃ = categorical([true, false, true, false, false, false, false], ordered=true),
        Y  = [1., 2., 3, 4, 5, 6, 7],
        W  = rand(7),
    )
    ps_lowerbound = 1e-8 
    weighted_fluctuation = false

    propensity_score_estimator = TMLE.JointConditionalDistributionEstimator(Dict(
        T => TMLE.MLConditionalDistributionEstimator(ConstantClassifier())
        for T in (:T₁, :T₂, :T₃)
    ))
    propensity_score_estimate = propensity_score_estimator(
        (TMLE.ConditionalDistribution(:T₁, [:W]), 
         TMLE.ConditionalDistribution(:T₂, [:W]), 
         TMLE.ConditionalDistribution(:T₃, [:W])),
        dataset,
        verbosity=0
    )

    cov, w = TMLE.clever_covariate_and_weights(Ψ, propensity_score_estimate, dataset;
        ps_lowerbound=ps_lowerbound, 
        weighted_fluctuation=weighted_fluctuation
    )
    @test cov ≈ [0, 8.575, -21.4375, 8.575, 0, -4.2875, -4.2875] atol=1e-3
    @test w == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
end

end

true