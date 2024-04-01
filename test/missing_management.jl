module TestMissingValues

using Test
using StableRNGs
using Random
using MLJLinearModels
using TMLE
using CategoricalArrays
using DataFrames

PKG_DIR = pkgdir(TMLE)

TEST_DIR = joinpath(PKG_DIR, "test")

include(joinpath(TEST_DIR, "helper_fns.jl"))

function dataset_with_missing_and_ordered_treatment(;n=1000)
    rng = StableRNG(123)
    W = rand(rng, n)
    T = rand(rng, [0, 1, 2], n)
    Y = T + 3W + randn(rng, n)
    dataset = DataFrame(W = W, T = categorical(T, ordered=true, levels=[0, 1, 2]), Y = Y)
    allowmissing!(dataset)
    dataset.W[1:5] .= missing
    dataset.T[6:10] .= missing
    dataset.Y[11:15] .= missing
    return dataset
end

@testset "Test nomissing" begin
    # If there is no missing data, this should be a no-op
    dataset = DataFrame(rand(10, 3), :auto)
    @test TMLE.nomissing(dataset) === dataset
    filtered = TMLE.nomissing(dataset, [:x1, :x2])
    # The only cost to this operation is the creation of a new table but the columns are indistinguishable
    @test filtered.x1 === dataset.x1
    @test filtered.x2 === dataset.x2

    # Now with missing values
    dataset = dataset_with_missing_and_ordered_treatment(;n=100)
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
    @test filtered.Y == dataset.Y[16:end]
end

@testset "Test estimation with missing values and ordered factor treatment" begin
    dataset = dataset_with_missing_and_ordered_treatment(;n=1000)
    Ψ = ATE(
        outcome=:Y, 
        treatment_values=(T=(case=1, control=0),),
        treatment_confounders=(T=[:W],))
    models=(Y=with_encoder(LinearRegressor()), T=LogisticClassifier(lambda=0))
    tmle = TMLEE(models=models, machine_cache=true)
    tmle_result, cache = tmle(Ψ, dataset; verbosity=0)
    test_coverage(tmle_result, 1)
    test_fluct_decreases_risk(cache)
end

end

true