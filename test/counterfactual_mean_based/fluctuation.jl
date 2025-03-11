module TestFluctuation

using Test
using TMLE
using MLJModels
using MLJBase
using DataFrames
using Distributions

@testset "Test Fluctuation with 1 Treatments" begin
    Ψ = ATE(
        outcome=:Y,
        treatment_confounders=(T=[:W₁, :W₂, :W₃],),
        treatment_values=(T=(case="a", control="b"),),
    )
    dataset = DataFrame(
        T = categorical(["a", "b", "c", "a", "a", "b", "a"]),
        Y = [1., 2., 3, 4, 5, 6, 7],
        W₁ = rand(7),
        W₂ = rand(7),
        W₃ = rand(7)
    )
    η = TMLE.CMRelevantFactors(
        TMLE.ConditionalDistribution(:Y, [:T, :W₁, :W₂, :W₃]),
        TMLE.ConditionalDistribution(:T, [:W₁, :W₂, :W₃])
    )
    η̂ = TMLE.CMRelevantFactorsEstimator(
        nothing,
        Dict(
            :Y => with_encoder(ConstantRegressor()), 
            :T => ConstantClassifier()
        )
    )
    η̂ₙ = η̂(η, dataset, verbosity = 0)
    X = dataset[!, collect(η̂ₙ.outcome_mean.estimand.parents)]
    y = dataset[!, η̂ₙ.outcome_mean.estimand.outcome]
    mse_initial = sum((TMLE.expected_value(η̂ₙ.outcome_mean, X) .- y).^2)

    # Weighted fluctuation
    weighted_fluctuation = TMLE.Fluctuation(Ψ, η̂ₙ; weighted=true)
    ## First check the initialization of the counterfactual/observed caches
    counterfactual_cache = TMLE.initialize_counterfactual_cache(weighted_fluctuation, X)
    expected_value = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0] # Constant predictions of the mean as per Q⁰
    @test mean.(counterfactual_cache.predictions[1]) == mean.(counterfactual_cache.predictions[2]) == expected_value
    @test counterfactual_cache.signs == [1., -1.]
    @test counterfactual_cache.covariates == [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    ]
    observed_cache = TMLE.initialize_observed_cache(weighted_fluctuation, X, y)
    @test observed_cache[:ŷ] isa Vector{<:Normal}
    @test observed_cache[:H] == [1.0, -1.0, 0.0, 1.0, 1.0, -1.0, 1.0] # This is used to fit, so weight has been removed
    @test observed_cache[:w] == [1.75, 3.5, 7., 1.75, 1.75, 3.5, 1.75] # weight is separate
    @test observed_cache[:y] isa Vector{Float64}
    ## Second fit the fluctuation
    w_machines, cache, w_report = MLJBase.fit(weighted_fluctuation, 0, X, y)
    ### Only one machine, only fitted the clever covariate 
    mach = only(w_machines)
    @test fitted_params(mach).features == [:covariate]
    ### Report entries
    @test length(w_report.epsilons) == length(w_report.estimates) == length(w_report.gradients) == 1
    @test w_report.epsilons[1][1] !== 0
    @test TMLE.hasconverged(only(w_report.gradients), 1e-10)
    ### Loss has decreased
    fluctuation_mean = TMLE.expected_value(MLJBase.predict(weighted_fluctuation, w_machines, X))
    mse_fluct = sum((fluctuation_mean .- y).^2)
    @test mse_fluct < mse_initial

    # Unweighted fluctuation
    unweighted_fluctuation = TMLE.Fluctuation(Ψ, η̂ₙ; weighted=false, tol=0, max_iter=3)
    ## First check the weight and covariates from the observed cache
    observed_cache = TMLE.initialize_observed_cache(unweighted_fluctuation, X, y)
    @test observed_cache[:H] == [1.75, -3.5, 0.0, 1.75, 1.75, -3.5, 1.75]
    @test observed_cache[:w] == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ## Second fit the fluctuation
    logs = [(:info, "TMLE step: 1."), (:info, "TMLE step: 2."), (:info, "TMLE step: 3."), (:info, "Convergence criterion not reached.")]
    uw_machines, cache, uw_report = @test_logs logs... MLJBase.fit(unweighted_fluctuation, 1, X, y);
    ### Only one machine, only fitted the clever covariate 
    @test length(uw_machines) == 3
    ### Report entries
    @test length(uw_report.epsilons) == length(uw_report.estimates) == length(uw_report.gradients) == 3
    @test uw_report.epsilons[1][1] !== w_report.epsilons[1][1] !== 0
    @test uw_report.epsilons[3][1] <= uw_report.epsilons[2][1] <= uw_report.epsilons[1][1]
    @test all(TMLE.hasconverged(gradient, 1e-10) for gradient in uw_report.gradients)
    ### Loss has not decreased
    fluctuation_mean = TMLE.expected_value(MLJBase.predict(unweighted_fluctuation, uw_machines, X))
    mse_fluct = sum((fluctuation_mean .- y).^2)
    @test mse_fluct < mse_initial 
end

@testset "Test Fluctuation with 2 Treatments" begin
    Ψ = AIE(
        outcome =:Y, 
        treatment_values=(
            T₁=(case=1, control=0), 
            T₂=(case=1, control=0)
        ),
        treatment_confounders = (
            T₁=[:W₁, :W₂, :W₃],
            T₂=[:W₁, :W₂, :W₃]
        )
    )
    dataset = DataFrame(
        T₁ = categorical([1, 0, 0, 1, 1, 1, 0]),
        T₂ = categorical([1, 1, 1, 1, 1, 0, 0]),
        Y = categorical([1, 1, 1, 1, 0, 0, 0]),
        W₁ = rand(7),
        W₂ = rand(7),
        W₃ = rand(7)
    )
    η = TMLE.CMRelevantFactors(
        TMLE.ConditionalDistribution(:Y, [:T₁, :T₂, :W₁, :W₂, :W₃]),
        (
            TMLE.ConditionalDistribution(:T₁, [:W₁, :W₂, :W₃]),
            TMLE.ConditionalDistribution(:T₂, [:W₁, :W₂, :W₃]),
        )
    )
    η̂ = TMLE.CMRelevantFactorsEstimator(
        nothing,
        Dict(
            :Y  => with_encoder(ConstantClassifier()), 
            :T₁ => ConstantClassifier(),
            :T₂ => ConstantClassifier()
        )
    )
    η̂ₙ = η̂(η, dataset, verbosity = 0)
    X = dataset[!, collect(η̂ₙ.outcome_mean.estimand.parents)]
    y = dataset[!, η̂ₙ.outcome_mean.estimand.outcome]

    fluctuation = TMLE.Fluctuation(Ψ, η̂ₙ; weighted=false)
    ## First check the initialization of the counterfactual/observed caches
    counterfactual_cache = TMLE.initialize_counterfactual_cache(fluctuation, X)
    @test length(counterfactual_cache.predictions) == 4 # (1, 1), (1, 0), (0, 1), (0, 0)
    for ŷ in counterfactual_cache.predictions
        Ey = TMLE.expected_value(ŷ)
        @test all(isapprox(x, 0.571, atol=1e-3) for x in Ey)
    end
    counterfactual_cache.signs == [1., 1, -1., -1]
    @test length(counterfactual_cache.covariates) == 4
    observed_cache = TMLE.initialize_observed_cache(fluctuation, X, y)
    @test observed_cache[:ŷ] isa UnivariateFiniteVector
    @test isapprox(observed_cache[:H], [2.44, -3.26, -3.26, 2.44, 2.44, -6.12, 8.16], atol=0.1)
    @test observed_cache[:w] == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    @test observed_cache[:y] isa Vector{Float64}

    machs, cache, report = MLJBase.fit(fluctuation, 0, X, y)
    @test TMLE.hasconverged(only(report.gradients), 1e-6)
end

@testset "Test same_type_nt" begin
    X = TMLE.same_type_nt([1., 2.], [1., 2])
    @test X.covariate isa Vector{Float64}
    @test X.offset isa Vector{Float64}

    X = TMLE.same_type_nt([1., 2.], [1.f0, 2.f0])
    @test X.covariate isa Vector{Float64}
    @test X.offset isa Vector{Float64}

    X = TMLE.same_type_nt([1.f0, 2.f0], [1., 2.])
    @test X.covariate isa Vector{Float32}
    @test X.offset isa Vector{Float32}
end

end

true