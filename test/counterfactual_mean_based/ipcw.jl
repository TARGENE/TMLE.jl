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

include(joinpath(dirname(dirname(pathof(TMLE))), "test", "helper_fns.jl"))

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

    # Test add_censoring_indicator
    df = DataFrame(Y = Union{Missing, Float64}[1.0, missing, 3.0, missing, 5.0])
    TMLE.add_censoring_indicator!(df, :Y)
    @test hasproperty(df, :Δ_Y)
    Δ = df[!, :Δ_Y]
    @test Δ isa CategoricalVector
    @test unwrap.(Δ) == [1, 0, 1, 0, 1]

    # The censoring indicator must register as binary so nuisance model selection treats it as a
    # classification target (see `is_binary` in nuisance_estimators.jl).
    @test TMLE.is_binary(df, :Δ_Y)
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

@testset "IPCW OSE and TMLE" begin
    dataset, ATE_true = ate_with_mar_missingness(n=5000)
    Ψ = ATE(
        outcome=:Y,
        treatment_values=(T=(case=1, control=0),),
        treatment_confounders=(T=[:W],)
    )
    ose = Ose()
    result, _ = ose(Ψ, dataset; verbosity=0)
    test_coverage(result, ATE_true)

    tmle = Tmle()
    result, _ = tmle(Ψ, dataset; verbosity=0)
    test_coverage(result, ATE_true)
end

"""
Generate a continuous outcome ATE problem with MAR missing *outcomes* AND some missing
*covariates*. Both W₁ and W₂ confound; W₂ is missing at random in a fraction of rows.
"""
function ate_with_missing_covariates(;n=5000)
    rng = StableRNG(42)
    W₁ = randn(rng, n)
    W₂ = randn(rng, n)
    T = rand(rng, n) .< LogExpFunctions.logistic.(0.5 .* W₁ .+ 0.3 .* W₂)
    Y = 2.0 .* T .+ W₁ .+ W₂ .+ 0.5 .* randn(rng, n)
    ATE_true = 2.0

    dataset = DataFrame(
        W₁ = W₁,
        W₂ = Vector{Union{Missing, Float64}}(W₂),
        T = categorical(Int.(T)),
        Y = Vector{Union{Missing, Float64}}(Y)
    )
    # MAR missing outcomes (depend on W₁)
    p_missing_Y = LogExpFunctions.logistic.(-1.0 .+ 0.8 .* W₁)
    # Missing covariate W₂ (depend on W₁, independent of the Y-missingness draw)
    p_missing_W₂ = LogExpFunctions.logistic.(-1.5 .+ 0.5 .* W₁)
    for i in 1:n
        rand(rng) < p_missing_Y[i] && (dataset.Y[i] = missing)
        rand(rng) < p_missing_W₂[i] && (dataset.W₂[i] = missing)
    end
    return dataset, ATE_true
end

@testset "CV-IPCW TMLE with missing covariates" begin
    dataset, ATE_true = ate_with_missing_covariates(n=5000)
    Ψ = ATE(
        outcome=:Y,
        treatment_values=(T=(case=1, control=0),),
        treatment_confounders=(T=[:W₁, :W₂],)
    )
    tmle = Tmle(resampling=CV(nfolds=3))
    result, _ = tmle(Ψ, dataset; verbosity=0)
    test_coverage(result, ATE_true)
end

@testset "CV-IPCW OSE with missing covariates" begin
    dataset, ATE_true = ate_with_missing_covariates(n=5000)
    Ψ = ATE(
        outcome=:Y,
        treatment_values=(T=(case=1, control=0),),
        treatment_confounders=(T=[:W₁, :W₂],)
    )
    ose = Ose(resampling=CV(nfolds=3))
    result, _ = ose(Ψ, dataset; verbosity=0)
    test_coverage(result, ATE_true)
end

function ate_with_treatment_dependent_missingness(;n=8000)
    rng = StableRNG(123)
    W = randn(rng, n)
    T = rand(rng, n) .< LogExpFunctions.logistic.(0.5 .* W)
    # Heterogeneous effect: 2 + 2.5W, so the ATE is 2 since E[W] = 0.
    Y = 2.0 .* T .+ W .+ 2.5 .* T .* W .+ 0.5 .* randn(rng, n)
    dataset = DataFrame(
        W = W,
        T = categorical(Int.(T)),
        Y = Vector{Union{Missing, Float64}}(Y)
    )
    p_missing = LogExpFunctions.logistic.(-0.5 .+ 1.2 .* W .+ 1.2 .* T)
    dataset[!, :Y][rand(rng, n) .< p_missing] .= missing
    return dataset, 2.0
end

@testset "Complete-case analysis is biased, IPCW is not" begin
    dataset, ATE_true = ate_with_treatment_dependent_missingness()
    Ψ = ATE(
        outcome=:Y,
        treatment_values=(T=(case=1, control=0),),
        treatment_confounders=(T=[:W],)
    )
    # Classic estimator: drop the censored rows, no censoring model at all.
    classic, _ = Tmle(ipcw=false)(Ψ, dataset; verbosity=0)
    lb, ub = confint(OneSampleTTest(classic))
    @test !(lb ≤ ATE_true ≤ ub)      # CI misses the truth entirely
    @test TMLE.estimate(classic) < 1 # severe downward bias (truth is 2)

    # IPCW estimator on the very same dataset, censored rows included.
    ipcw, _ = Tmle()(Ψ, dataset; verbosity=0)
    test_coverage(ipcw, ATE_true)
end

end

true
