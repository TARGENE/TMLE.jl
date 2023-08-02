module TestFitNuisance

using Test
using TMLE
using StableRNGs
using Distributions
using LogExpFunctions
using Random
using DataFrames
using CategoricalArrays
using MLJLinearModels
using MLJBase

function make_dataset(;n=100, rng=StableRNG(123))
    μy_cont(W₁, W₂, W₃, T₁) = 2W₁ .- 4W₂ .- 3W₃ .+ 2T₁
    μy_bin(W₁, W₂, W₃, T₁) = logistic.(μy_cont(W₁, W₂, W₃, T₁))
    W₁ = rand(rng, Normal(), n)
    W₂ = rand(rng, Normal(), n)
    W₃ = rand(rng, Normal(), n)
    μT₁ = logistic.(0.3W₁ .- 1.4W₂ .+ 1.9W₃)
    T₁ = rand(rng, Uniform(), n) .< μT₁
    Ybin = rand(rng, Uniform(), n) .< μy_bin(W₁, W₂, W₃, T₁)
    Ycont = μy_cont(W₁, W₂, W₃, T₁) + rand(rng, Normal(), n)
    return DataFrame(
        W₁ = W₁,
        W₂ = W₂,
        W₃ = W₃,
        T₁ = categorical(T₁),
        Ybin = categorical(Ybin),
        Ycont = Ycont
    )
end

@testset "Test fit_nuisance! binary Y" begin
    dataset = make_dataset(;n=50_000, rng=StableRNG(123))
    cache = TMLECache(dataset)
    Ψ = ATE(
        outcome=:Ybin,
        treatment=(T₁=(case=true, control=false),),
        confounders = [:W₁, :W₂, :W₃]
    )
    η_spec = NuisanceSpec(
        LogisticClassifier(lambda=0.),
        LogisticClassifier(lambda=0.)
    )
    update!(cache, Ψ, η_spec)
    TMLE.fit_nuisance!(cache, verbosity=0)
    # Check G fit
    Gcoefs = fitted_params(cache.η.G).coefs
    @test Gcoefs[1][2] ≈ 0.298 atol=0.001 # outcome is 0.3
    @test Gcoefs[2][2] ≈ -1.441 atol=0.001 # outcome is -1.4
    @test Gcoefs[3][2] ≈ 1.953 atol=0.001 # outcome is 1.9
    # Check Q fit
    Qcoefs = fitted_params(cache.η.Q).coefs
    @test Qcoefs[1][2] ≈ 2.000 atol=0.001 # outcome is 2
    @test Qcoefs[2][2] ≈ -4.006 atol=0.001 # outcome is -4
    @test Qcoefs[3][2] ≈ -3.046 atol=0.001 # outcome is -3
    @test Qcoefs[4][2] ≈ -2.062 atol=0.001 # outcome is 2 on T₁=1 but T₁=0 is encoded
    @test fitted_params(cache.η.Q).intercept ≈ 2.053 atol=0.001

    # Fluctuate the inital Q, the log_loss should decrease
    TMLE.tmle_step!(cache; verbosity=0, threshold=1e-8)
    ll_fluct = mean(log_loss(cache.data[:Qfluct], dataset.Ybin))
    ll_init = mean(log_loss(cache.data[:Q₀], dataset.Ybin))
    @test ll_fluct < ll_init
end


@testset "Test fit_nuisance! continuous Y" begin
    dataset = make_dataset(;n=50_000, rng=StableRNG(123))
    cache = TMLECache(dataset)
    Ψ = ATE(
        outcome=:Ycont,
        treatment=(T₁=(case=true, control=false),),
        confounders = [:W₁, :W₂, :W₃]
    )
    η_spec = NuisanceSpec(
        LinearRegressor(),
        LogisticClassifier(lambda=0.)
    )
    update!(cache, Ψ, η_spec)
    TMLE.fit_nuisance!(cache, verbosity=0)
    # Check G fit
    Gcoefs = fitted_params(cache.η.G).coefs
    @test Gcoefs[1][2] ≈ 0.298 atol=0.001 # outcome is 0.3
    @test Gcoefs[2][2] ≈ -1.441 atol=0.001 # outcome is -1.4
    @test Gcoefs[3][2] ≈ 1.953 atol=0.001 # outcome is 1.9
    # Check Q fit
    Qcoefs = fitted_params(cache.η.Q).coefs
    @test Qcoefs[1][2] ≈ 2.001 atol=0.001 # outcome is 2
    @test Qcoefs[2][2] ≈ -4.002 atol=0.001 # outcome is -4
    @test Qcoefs[3][2] ≈ -2.999 atol=0.001 # outcome is -3
    @test Qcoefs[4][2] ≈ -1.988 atol=0.001 # outcome is 2 on T₁=1 but T₁=0 is encoded
    @test fitted_params(cache.η.Q).intercept ≈ 1.996 atol=0.001

    # Fluctuate the inital Q, the RMSE should decrease
    TMLE.tmle_step!(cache; verbosity=0, threshold=1e-8)
    rmse_fluct = rmse(mean.(cache.data[:Qfluct]), dataset.Ycont)
    rmse_init = rmse(cache.data[:Q₀], dataset.Ycont)
    @test rmse_fluct < rmse_init
end

end

true