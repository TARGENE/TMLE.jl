module TestCache

using Test
using TMLE
using MLJLinearModels
using MLJModels
using CategoricalArrays
using MLJBase
using StableRNGs
using Distributions

function naive_dataset(;n=100)
    rng = StableRNG(123)
    W = rand(rng, Uniform(), n)
    T = rand(rng, [0, 1], n)
    y = 3W .+ T .+ T.*W + rand(rng, Normal(0, 0.05), n)
    return (
        y = y,
        W = W,
        T = categorical(T)
    )
end


function fakefill!(cache)
    n = 10
    X = (x=rand(n),)
    y = rand(n)
    mach = machine(LinearRegressor(), X, y)
    cache.η = TMLE.NuisanceEstimands(mach, mach, mach, mach)
end

function fakecache()
    n = 10
    dataset = (
        W₁=vcat(missing, rand(n-1)),
        W₂=vcat([missing, missing], rand(n-2)),
        C₁=rand(n),
        T₁=rand([1, 0], n),
        T₂=rand([1, 0], n),
        T₃=["CT", "CC", "TT", "TT", "TT", "CT", "CC", "TT", "CT", "CC"],
        y=rand(n),
        ynew=vcat([missing, missing, missing], rand(n-3))
        )
    cache = TMLE.TMLECache(dataset)
    return cache
end

@testset "Test check_treatment_values" begin
    cache = fakecache()
    Ψ = ATE(
        target=:y,
        treatment=(T₁=(case=1, control=0),),
        confounders=[:W₁],
    )
    TMLE.check_treatment_values(cache, Ψ)
    Ψ = ATE(
        target=:y,
        treatment=(
            T₁=(case=1, control=0), 
            T₂=(case=true, control=false),),
        confounders=[:W₁],
    )
    @test_throws ArgumentError(string("The 'case' string representation: 'true' for treatment",
            " T₂ in Ψ does not match any level of the corresponding variable",
            " in the dataset: [\"0\", \"1\"]")) TMLE.check_treatment_values(cache, Ψ)

    Ψ = CM(
        target=:y,
        treatment=(T₃="CT",),
        confounders=[:W₁],
    )
    TMLE.check_treatment_values(cache, Ψ)
end

@testset "Test update!(cache, Ψ)" begin
    cache = fakecache()
    @test !isdefined(cache, :Ψ)
    @test !isdefined(cache, :η_spec)

    Ψ = ATE(
        target=:y,
        treatment=(T₁=(case=1, control=0),),
        confounders=[:W₁],
        covariates=[:C₁]
    )
    η_spec = NuisanceSpec(
        RidgeRegressor(),
        LogisticClassifier(lambda=0)
    )
    TMLE.update!(cache, Ψ, η_spec)
    @test cache.Ψ == Ψ
    @test cache.η_spec == η_spec
    @test cache.η.Q === nothing
    @test cache.η.H === nothing
    @test cache.η.G === nothing
    @test cache.η.F === nothing
    # Since no missing data is present, the columns are just propagated
    # to the no_missing dataset in the sense of ===
    for colname in keys(cache.data[:no_missing])
        cache.data[:no_missing][colname] === cache.data[:source][colname]
    end
    # Pretend the cache nuisance estimands have been estimated
    fakefill!(cache)
    @test cache.η.H !== nothing
    @test cache.η.G !== nothing
    @test cache.η.Q !== nothing
    @test cache.η.F !== nothing
    # New treatment configuration and new confounders set
    # Nuisance estimands can be reused
    Ψ = ATE(
        target=:y,
        treatment=(T₁=(case=0, control=1),),
        confounders=[:W₁, :W₂],
        covariates=[:C₁]
    )
    TMLE.update!(cache, Ψ)
    @test cache.η.H !== nothing
    @test cache.η.G === nothing
    @test cache.η.Q === nothing
    @test cache.η.F === nothing
    @test cache.Ψ == Ψ
    for colname in keys(cache.data[:no_missing])
        @test length(cache.data[:no_missing][colname]) == 8
    end

    # Change the target
    # E[Y|X] must be refit
    fakefill!(cache)
    Ψ = ATE(
        target=:ynew,
        treatment=(T₁=(case=0, control=1),),
        confounders=[:W₁, :W₂],
        covariates=[:C₁]
    )
    TMLE.update!(cache, Ψ)
    @test cache.η.H !== nothing
    @test cache.η.G !== nothing
    @test cache.η.Q === nothing
    @test cache.η.F === nothing
    @test cache.Ψ == Ψ
    for colname in keys(cache.data[:no_missing])
        @test length(cache.data[:no_missing][colname]) == 7
    end

    # Change the covariate
    # E[Y|X] must be refit
    fakefill!(cache)
    Ψ = ATE(
        target=:ynew,
        treatment=(T₁=(case=0, control=1),),
        confounders=[:W₁, :W₂],
    )
    TMLE.update!(cache, Ψ)
    @test cache.η.H !== nothing
    @test cache.η.G !== nothing
    @test cache.η.Q === nothing
    @test cache.η.F === nothing
    @test cache.Ψ == Ψ

    # Change only treatment setting
    # only F needs refit
    fakefill!(cache)
    Ψ = ATE(
        target=:ynew,
        treatment=(T₁=(case=1, control=0),),
        confounders=[:W₁, :W₂],
    )
    TMLE.update!(cache, Ψ)
    @test cache.η.H !== nothing
    @test cache.η.G !== nothing
    @test cache.η.Q !== nothing
    @test cache.η.F === nothing
    @test cache.Ψ == Ψ

    # Change the treatments
    # all η must be refit
    fakefill!(cache)
    Ψ = ATE(
        target=:ynew,
        treatment=(T₂=(case=1, control=0),),
        confounders=[:W₁, :W₂],
    )
    TMLE.update!(cache, Ψ)
    @test cache.η.H === nothing
    @test cache.η.G === nothing
    @test cache.η.Q === nothing
    @test cache.η.F === nothing
    @test cache.Ψ == Ψ
end

@testset "Test update!(cache, η_spec)" begin
    cache = fakecache()
    Ψ = ATE(
        target=:y,
        treatment=(T₁=(case=1, control=0),),
        confounders=[:W₁],
        covariates=[:C₁]
    )
    η_spec = NuisanceSpec(
        RidgeRegressor(),
        LogisticClassifier(lambda=0)
    )
    TMLE.update!(cache, Ψ, η_spec)
    fakefill!(cache)
    # Change the Q learning stategy
    η_spec = NuisanceSpec(
        RidgeRegressor(lambda=10),
        LogisticClassifier(lambda=0)
    )
    TMLE.update!(cache, η_spec)
    @test cache.η.H !== nothing
    @test cache.η.G !== nothing
    @test cache.η.Q === nothing
    @test cache.η.F === nothing
    @test cache.η_spec == η_spec

    # Change the G learning stategy
    fakefill!(cache)
    η_spec = NuisanceSpec(
        RidgeRegressor(lambda=10),
        LogisticClassifier(lambda=10)
    )
    TMLE.update!(cache, η_spec)
    @test cache.η.H !== nothing
    @test cache.η.G === nothing
    @test cache.η.Q !== nothing
    @test cache.η.F === nothing
    @test cache.η_spec == η_spec
end

@testset "Test counterfactual_aggregate" begin
    n=100
    dataset = naive_dataset(;n=n)
    Ψ = ATE(
        target = :y,
        treatment = (T=(case=1, control=0),),
        confounders = [:W]
    )
    η_spec = NuisanceSpec(
        MLJModels.ConstantRegressor(),
        ConstantClassifier()
    )
    # Nuisance estimand estimation
    tmle_result, cache = tmle(Ψ, η_spec, dataset; verbosity=0);
    # The inital estimator for Q is a constant predictor, 
    # its  prediction is the same for each counterfactual
    # and the initial aggregate is zero.
    # Since G is also constant the output after fluctuating 
    # must be constant over the counterfactual dataset
    counterfactual_aggregate, counterfactual_aggregateᵢ = TMLE.counterfactual_aggregates(cache; threshold=1e-8)
    @test counterfactual_aggregateᵢ == zeros(n)
    @test all(first(counterfactual_aggregate) .== counterfactual_aggregate)

    # Replacing Q with a linear regression
    η_spec = NuisanceSpec(
        LinearRegressor(),
        ConstantClassifier()
    )
    tmle!(cache, η_spec, verbosity=0);
    X₁ = TMLE.Qinputs(cache.η.H, (W=dataset.W, T=categorical(ones(Int, n), levels=levels(dataset.T))), Ψ)
    X₀ = TMLE.Qinputs(cache.η.H, (W=dataset.W, T=categorical(zeros(Int, n), levels=levels(dataset.T))), Ψ)
    ŷ₁ =  TMLE.expected_value(MLJBase.predict(cache.η.Q, X₁))
    ŷ₀ = TMLE.expected_value(MLJBase.predict(cache.η.Q, X₀))
    expected_cf_agg = ŷ₁ - ŷ₀
    counterfactual_aggregate, counterfactual_aggregateᵢ = TMLE.counterfactual_aggregates(cache; threshold=1e-8)
    @test counterfactual_aggregateᵢ == expected_cf_agg
    # This is the coefficient in the linear regression model
    var, coef = fitted_params(cache.η.Q).coefs[2]
    @test var == :T__0
    @test all(coef ≈ -x for x ∈ counterfactual_aggregateᵢ)

end

end

true