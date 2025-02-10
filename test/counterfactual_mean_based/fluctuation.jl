module TestFluctuation

using Test
using TMLE
using MLJModels
using MLJBase
using DataFrames

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

    ps_lowerbound = 1e-8
    # Weighted fluctuation
    expected_weights = [1.75, 3.5, 7., 1.75, 1.75, 3.5, 1.75]
    expected_covariate = [1., -1., 0.0, 1., 1., -1., 1.]
    fluctuation = TMLE.Fluctuation(Ψ, η̂ₙ;
        tol=nothing,
        max_iter=1,
        ps_lowerbound=ps_lowerbound,
        weighted=true,
        cache=true    
    )
    ## First check the initialization of the counterfactual cache used to 
    ## compute gradients and estimates
    counterfactual_cache = TMLE.initialize_counterfactual_cache(Ψ, η̂ₙ.outcome_mean, η̂ₙ.propensity_score, X; ps_lowerbound=ps_lowerbound)
    @test counterfactual_cache.predictions == [
        [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
        [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
    ] # Constant predictions of the mean as per Q⁰
    @test counterfactual_cache.signs == [1., -1.]
    @test counterfactual_cache.covariates == [
        [1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75],
        [-3.5, -3.5, -3.5, -3.5, -3.5, -3.5, -3.5]
    ] # Covariates include the weights since this is for evaluation, not training
    fitresult, cache, report = MLJBase.fit(fluctuation, 0, X, y)
    report = only(report)
    @test report.epsilon isa Vector{<:AbstractFloat}
    @test report.gradient isa Vector{<:AbstractFloat}
    @test length(report.gradient) == 7
    @test report.estimate isa AbstractFloat
    fluctuation_mean = TMLE.expected_value(MLJBase.predict(fluctuation, fitresult, X))
    mse_fluct = sum((fluctuation_mean .- y).^2)
    @test mse_fluct < mse_initial
    mach = only(fitresult.machines)
    @test fitted_params(mach).features == [:covariate]
    Xfluct, weights = TMLE.clever_covariate_offset_and_weights(fluctuation, X)
    @test weights == expected_weights
    @test mach.data[3] == expected_weights
    @test mach.data[1].covariate == expected_covariate

    # Unweighted fluctuation
    fluctuation.weighted = false
    expected_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    expected_covariate = [1.75, -3.5, 0.0, 1.75, 1.75, -3.5, 1.75]
    fitresult, cache, report = MLJBase.fit(fluctuation, 0, X, y)
    mach = only(fitresult.machines)
    @test mach.data[3] == expected_weights
    Xfluct, weights = TMLE.clever_covariate_offset_and_weights(fluctuation, X)
    @test weights == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    @test Xfluct.covariate == expected_covariate
    @test mach.data[1].covariate == expected_covariate
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

    fluctuation = TMLE.Fluctuation(Ψ, η̂ₙ;
        tol=nothing,
        max_iter=1,
        ps_lowerbound=1e-8,
        weighted=false,   
        cache=true     
    )
    fitresult, cache, report = MLJBase.fit(fluctuation, 0, X, y)
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