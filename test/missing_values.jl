module TestMissingValues

using Test
using StableRNGs
using Random
using MLJLinearModels
using TMLE
using CategoricalArrays
using DataFrames

function dataset_with_missing(;n=1000)
    rng = StableRNG(123)
    W = rand(rng, n)
    T = rand(rng, [0, 1], n)
    y = T + 3W + randn(rng, n)
    dataset = DataFrame(W = W, T = categorical(T), y = y)
    allowmissing!(dataset)
    dataset.W[1:5] .= missing
    dataset.T[6:10] .= missing
    dataset.y[11:15] .= missing
    return dataset
end

@testset "Test nomissing" begin
    dataset = dataset_with_missing(;n=100)
    # filter missing rows based on W column
    filtered = TMLE.nomissing(dataset, [:W])
    @test filtered.W == dataset.W[6:end]
    # filter missing rows based on W, T columns
    filtered = TMLE.nomissing(dataset, [:W, :T])
    @test filtered.W == dataset.W[11:end]
    @test filtered.T == dataset.T[11:end]
    # filter all missing rows
    filtered = TMLE.nomissing(dataset)
    @test filtered.W == dataset.W[16:end]
    @test filtered.T == dataset.T[16:end]
    @test filtered.y == dataset.y[16:end]
end

@testset "Test missing value dataset" begin
    dataset = dataset_with_missing(;n=1000)
    Ψ = ATE(
        target=:y, 
        confounders=[:W], 
        treatment=(T=(case=1, control=0),))
    η_spec = NuisanceSpec(LinearRegressor(), LogisticClassifier(lambda=0))
    tmle_results, initial_results, cache = tmle(Ψ, η_spec, dataset; verbosity=0)
    @test estimate(tmle_results) ≈ 0.986 atol=1e-3
end

end

true