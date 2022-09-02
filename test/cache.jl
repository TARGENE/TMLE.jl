module TestCache

using Test
using TMLE
using MLJLinearModels
using MLJBase

function fakefill!(cache)
    n = 10
    X = (x=rand(n),)
    y = rand(n)
    mach = machine(LinearRegressor(), X, y)
    cache.η = TMLE.NuisanceParameters(mach, mach, mach, mach)
end

function fakecache()
    Ψ = ATE(
        target=:y,
        treatment=(T₁=(case=1, control=0),),
        confounders=[:W₁, :W₂],
        covariates=[:C₁]
    )
    η_spec = (
        Q = RidgeRegressor(),
        G = LogisticClassifier(lambda=0)
    )
    dataset = (;)
    cache = TMLE.TMLECache(Ψ, η_spec, dataset)
    fakefill!(cache)
    return cache
end


@testset "Test update!(cache, Ψ)" begin
    # Same variables, new treatment configuration
    # Nuisance parameters can be reused
    cache = fakecache()
    Ψ = ATE(
        target=:y,
        treatment=(T₁=(case=0, control=1),),
        confounders=[:W₁, :W₂],
        covariates=[:C₁]
    )
    TMLE.update!(cache, Ψ)
    @test cache.η.H !== nothing
    @test cache.η.G !== nothing
    @test cache.η.Q !== nothing
    @test cache.η.F === nothing
    @test cache.Ψ == Ψ

    # Change the target
    # E[Y|X] must be refit
    cache = fakecache()
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

    # Change the covariate
    # E[Y|X] must be refit
    cache = fakecache()
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

    # Change the confounders
    # G and Q must be refit
    cache = fakecache()
    Ψ = ATE(
        target=:ynew,
        treatment=(T₁=(case=0, control=1),),
        confounders=[:W₁],
    )
    TMLE.update!(cache, Ψ)
    @test cache.η.H !== nothing
    @test cache.η.G === nothing
    @test cache.η.Q === nothing
    @test cache.η.F === nothing
    @test cache.Ψ == Ψ

    # Change the treatments
    # all η must be refit
    cache = fakecache()
    Ψ = ATE(
        target=:ynew,
        treatment=(T₂=(case=0, control=1),),
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
    # Change the Q learning stategy
    cache = fakecache()
    η_spec = (
        Q = RidgeRegressor(lambda=10),
        G = LogisticClassifier(lambda=0)
    )
    TMLE.update!(cache, η_spec)
    @test cache.η.H !== nothing
    @test cache.η.G !== nothing
    @test cache.η.Q === nothing
    @test cache.η.F === nothing
    @test cache.η_spec == η_spec

    # Change the G learning stategy
    cache = fakecache()
    η_spec = (
        Q = RidgeRegressor(),
        G = LogisticClassifier(lambda=10)
    )
    TMLE.update!(cache, η_spec)
    @test cache.η.H !== nothing
    @test cache.η.G === nothing
    @test cache.η.Q !== nothing
    @test cache.η.F === nothing
    @test cache.η_spec == η_spec
end

end

true