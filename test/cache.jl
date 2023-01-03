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
        confounders=[:W₁],
        covariates=[:C₁]
    )
    η_spec = NuisanceSpec(
        RidgeRegressor(),
        LogisticClassifier(lambda=0)
    )
    n = 10
    dataset = (
        W₁=vcat(missing, rand(n-1)),
        W₂=vcat([missing, missing], rand(n-2)),
        C₁=rand(n),
        T₁=rand([1, 0], n),
        T₂=rand([1, 0], n),
        y=rand(n),
        ynew=vcat([missing, missing, missing], rand(n-3))
        )
    cache = TMLE.TMLECache(Ψ, η_spec, dataset, false)
    fakefill!(cache)
    return cache
end


@testset "Test update!(cache, Ψ)" begin
    cache = fakecache()
    # Since no missing data is present, the columns are just propagated
    # to the no_missing dataset in the sense of ===
    for colname in keys(cache.dataset[:no_missing])
        cache.dataset[:no_missing][colname] === cache.dataset[:source][colname]
    end
    # New treatment configuration and new confounders set
    # Nuisance parameters can be reused
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
    for colname in keys(cache.dataset[:no_missing])
        @test length(cache.dataset[:no_missing][colname]) == 8
    end

    # Change the target
    # E[Y|X] must be refit
    cache = fakecache()
    Ψ = ATE(
        target=:ynew,
        treatment=(T₁=(case=0, control=1),),
        confounders=[:W₁],
        covariates=[:C₁]
    )
    TMLE.update!(cache, Ψ)
    @test cache.η.H !== nothing
    @test cache.η.G !== nothing
    @test cache.η.Q === nothing
    @test cache.η.F === nothing
    @test cache.Ψ == Ψ
    for colname in keys(cache.dataset[:no_missing])
        @test length(cache.dataset[:no_missing][colname]) == 7
    end

    # Change the covariate
    # E[Y|X] must be refit
    cache = fakecache()
    Ψ = ATE(
        target=:y,
        treatment=(T₁=(case=0, control=1),),
        confounders=[:W₁],
    )
    TMLE.update!(cache, Ψ)
    @test cache.η.H !== nothing
    @test cache.η.G !== nothing
    @test cache.η.Q === nothing
    @test cache.η.F === nothing
    @test cache.Ψ == Ψ

    # Change only treatment setting
    # only F needs refit
    cache = fakecache()
    Ψ = ATE(
        target=:y,
        treatment=(T₁=(case=0, control=1),),
        confounders=[:W₁],
        covariates=[:C₁]
    )
    TMLE.update!(cache, Ψ)
    @test cache.η.H !== nothing
    @test cache.η.G !== nothing
    @test cache.η.Q !== nothing
    @test cache.η.F === nothing
    @test cache.Ψ == Ψ

    # Change the treatments
    # all η must be refit
    cache = fakecache()
    Ψ = ATE(
        target=:y,
        treatment=(T₂=(case=0, control=1),),
        confounders=[:W₁],
        covariates=[:C₁]
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
    cache = fakecache()
    η_spec = NuisanceSpec(
        RidgeRegressor(),
        LogisticClassifier(lambda=10)
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