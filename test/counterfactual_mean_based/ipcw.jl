module TestIPCW

using Test
using StableRNGs
using Random
using Distributions
using DataFrames
using CategoricalArrays
using MLJBase
using LogExpFunctions
using TMLE

PKG_DIR = pkgdir(TMLE)
TEST_DIR = joinpath(PKG_DIR, "test")
include(joinpath(TEST_DIR, "helper_fns.jl"))

@testset "Censoring indicator utilities" begin
    # Test censoring_indicator_name
    @test TMLE.censoring_indicator_name(:Y) == :Δ_Y
    @test TMLE.censoring_indicator_name(:outcome) == :Δ_outcome

    # Test has_missing_outcomes
    df = DataFrame(Y = [1.0, 2.0, 3.0], X = [1, 2, 3])
    @test TMLE.has_missing_outcomes(df, :Y) == false
    allowmissing!(df, :Y)
    df.Y[2] = missing
    @test TMLE.has_missing_outcomes(df, :Y) == true
    @test TMLE.has_missing_outcomes(df, :X) == false

    # Test add_censoring_indicator!
    df = DataFrame(Y = Union{Missing, Float64}[1.0, missing, 3.0, missing, 5.0])
    TMLE.add_censoring_indicator!(df, :Y)
    @test hasproperty(df, :Δ_Y)
    Δ = df[!, :Δ_Y]
    @test Δ isa CategoricalVector
    @test unwrap.(Δ) == [1, 0, 1, 0, 1]

    # Test get_censoring_indicator
    Δ_float = TMLE.get_censoring_indicator(df, :Y)
    @test Δ_float == [1.0, 0.0, 1.0, 0.0, 1.0]
    @test eltype(Δ_float) == Float64
end

@testset "CMRelevantFactors with censoring_score" begin
    om = TMLE.ConditionalDistribution(:Y, (:T, :W))
    ps = TMLE.ConditionalDistribution(:T, (:W,))
    cs = TMLE.ConditionalDistribution(:Δ_Y, (:T, :W))

    # Without censoring
    rf = TMLE.CMRelevantFactors(om, (ps,))
    @test rf.censoring_score === nothing

    # With censoring
    rf_ipcw = TMLE.CMRelevantFactors(om, (ps,), cs)
    @test rf_ipcw.censoring_score === cs
    @test :Δ_Y ∈ TMLE.variables(rf_ipcw)

    # Keyword constructor
    rf_kw = TMLE.CMRelevantFactors(outcome_mean=om, propensity_score=(ps,), censoring_score=cs)
    @test rf_kw == rf_ipcw
end

@testset "get_relevant_factors with IPCW" begin
    # No dataset → no censoring
    Ψ = ATE(
        outcome=:Y,
        treatment_values=(T=(case=1, control=0),),
        treatment_confounders=(T=[:W],)
    )
    rf = TMLE.get_relevant_factors(Ψ)
    @test rf.censoring_score === nothing

    # Dataset without missing → no censoring
    df = DataFrame(Y=randn(10), T=categorical(rand(0:1, 10)), W=randn(10))
    rf = TMLE.get_relevant_factors(Ψ; dataset=df)
    @test rf.censoring_score === nothing

    # Dataset with missing outcome → censoring activated
    allowmissing!(df, :Y)
    df.Y[1] = missing
    rf = TMLE.get_relevant_factors(Ψ; dataset=df)
    @test rf.censoring_score !== nothing
    @test rf.censoring_score.outcome == :Δ_Y
end

"""
Generate a continuous outcome ATE problem with MAR missingness.
Missingness depends on W (confounders) creating a MAR pattern.
"""
function ate_with_mar_missingness(;n=2000)
    rng = StableRNG(42)
    W = randn(rng, n)
    T = rand(rng, n) .< LogExpFunctions.logistic.(0.5 .* W)
    # True ATE = 2
    Y = 2.0 .* T .+ W .+ 0.5 .* randn(rng, n)
    ATE_true = 2.0

    dataset = DataFrame(
        W = W,
        T = categorical(Int.(T)),
        Y = Vector{Union{Missing, Float64}}(Y)
    )
    # MAR missingness: P(missing | W) depends on W
    p_missing = LogExpFunctions.logistic.(-1.0 .+ 0.8 .* W)
    for i in 1:n
        if rand(rng) < p_missing[i]
            dataset.Y[i] = missing
        end
    end
    return dataset, ATE_true
end

"""
Generate a continuous outcome ATE problem with MCAR missingness.
"""
function ate_with_mcar_missingness(;n=2000)
    rng = StableRNG(42)
    W = randn(rng, n)
    T = rand(rng, n) .< LogExpFunctions.logistic.(0.5 .* W)
    # True ATE = 2
    Y = 2.0 .* T .+ W .+ 0.5 .* randn(rng, n)
    ATE_true = 2.0

    dataset = DataFrame(
        W = W,
        T = categorical(Int.(T)),
        Y = Vector{Union{Missing, Float64}}(Y)
    )
    # MCAR: 20% random missingness
    for i in 1:n
        if rand(rng) < 0.2
            dataset.Y[i] = missing
        end
    end
    return dataset, ATE_true
end

@testset "IPCW OSE: MCAR missingness" begin
    dataset, ATE_true = ate_with_mcar_missingness(n=5000)
    Ψ = ATE(
        outcome=:Y,
        treatment_values=(T=(case=1, control=0),),
        treatment_confounders=(T=[:W],)
    )
    ose = Ose()
    result, _ = ose(Ψ, dataset; verbosity=0)
    test_coverage(result, ATE_true)
end

@testset "IPCW TMLE: MCAR missingness" begin
    dataset, ATE_true = ate_with_mcar_missingness(n=5000)
    Ψ = ATE(
        outcome=:Y,
        treatment_values=(T=(case=1, control=0),),
        treatment_confounders=(T=[:W],)
    )
    tmle = Tmle()
    result, _ = tmle(Ψ, dataset; verbosity=0)
    test_coverage(result, ATE_true)
end

@testset "IPCW OSE: MAR missingness" begin
    dataset, ATE_true = ate_with_mar_missingness(n=5000)
    Ψ = ATE(
        outcome=:Y,
        treatment_values=(T=(case=1, control=0),),
        treatment_confounders=(T=[:W],)
    )
    ose = Ose()
    result, _ = ose(Ψ, dataset; verbosity=0)
    test_coverage(result, ATE_true)
end

@testset "IPCW TMLE: MAR missingness" begin
    dataset, ATE_true = ate_with_mar_missingness(n=5000)
    Ψ = ATE(
        outcome=:Y,
        treatment_values=(T=(case=1, control=0),),
        treatment_confounders=(T=[:W],)
    )
    tmle = Tmle()
    result, _ = tmle(Ψ, dataset; verbosity=0)
    test_coverage(result, ATE_true)
end

@testset "No missingness: censoring_score is nothing" begin
    # When no missing data, IPCW should not be triggered
    rng = StableRNG(123)
    n = 500
    W = randn(rng, n)
    T = categorical(Int.(rand(rng, n) .< 0.5))
    Y = 2.0 .* Float64.(unwrap.(T)) .+ W .+ randn(rng, n)
    dataset = DataFrame(W=W, T=T, Y=Y)

    Ψ = ATE(
        outcome=:Y,
        treatment_values=(T=(case=1, control=0),),
        treatment_confounders=(T=[:W],)
    )
    tmle = Tmle()
    result, cache = tmle(Ψ, dataset; verbosity=0)
    # Verify censoring_score is nothing
    targeted_factors = cache[:targeted_factors]
    @test targeted_factors.censoring_score === nothing
    # Verify no censoring indicator column added
    @test !hasproperty(dataset, :Δ_Y)
    test_coverage(result, 2.0)
end

end

true
