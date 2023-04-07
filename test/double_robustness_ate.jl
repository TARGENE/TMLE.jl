module TestDoubleRobustnessATE

using TMLE
using Random
using Test
using Distributions
using MLJBase
using MLJLinearModels
using MLJModels
using StableRNGs
using StatsBase
using HypothesisTests
using LogExpFunctions

include("helper_fns.jl")

"""
Q and G are two logistic models
"""
function binary_target_binary_treatment_pb(;n=100)
    rng = StableRNG(123)
    p_w() = 0.3
    pa_given_w(w) = 1 ./ (1 .+ exp.(-0.5w .+ 1))
    py_given_aw(a, w) = 1 ./ (1 .+ exp.(2w .- 3a .+ 1))
    # Sample from dataset
    Unif = Uniform(0, 1)
    w = rand(rng, Unif, n) .< p_w()
    t = rand(rng, Unif, n) .< pa_given_w(w)
    y = rand(rng, Unif, n) .< py_given_aw(t, w)
    # Convert to dataframe to respect the Tables.jl
    # and convert types
    W = convert(Array{Float64}, w)
    T = categorical(t)
    y = categorical(y)
    # Compute the theoretical ATE
    ATE₁ = py_given_aw(1, 1)*p_w() + (1-p_w())*py_given_aw(1, 0)
    ATE₀ = py_given_aw(0, 1)*p_w() + (1-p_w())*py_given_aw(0, 0)
    ATE = ATE₁ - ATE₀
    
    return (T=T, W=W, y=y), ATE
end

"""
From https://www.degruyter.com/document/doi/10.2202/1557-4679.1043/html
The theoretical ATE is 1
"""
function continuous_target_binary_treatment_pb(;n=100)
    rng = StableRNG(123)
    Unif = Uniform(0, 1)
    W = float(rand(rng, Bernoulli(0.5), n, 3))
    W₁, W₂, W₃ = W[:, 1], W[:, 2], W[:, 3]
    t = rand(rng, Unif, n) .< logistic.(0.5W₁ + 1.5W₂ - W₃)
    y = 4t + 25W₁ + 3W₂ - 4W₃ + rand(rng, Normal(0, 0.1), n)
    # Type coercion
    T = categorical(t)
    return (T = T, W₁ = W₁, W₂ = W₂, W₃ = W₃, y = y), 4
end

function continuous_target_categorical_treatment_pb(;n=100, control="TT", case="AA")
    rng = StableRNG(123)
    ft(T) = (T .== "AA") - (T .== "AT") + 2(T .== "TT")
    fw(W₁, W₂, W₃) = 2W₁ + 3W₂ - 4W₃

    W = float(rand(rng, Bernoulli(0.5), n, 3))
    W₁, W₂, W₃ = W[:, 1], W[:, 2], W[:, 3]
    θ = rand(rng, 3, 3)
    softmax = exp.(W*θ) ./ sum(exp.(W*θ), dims=2)
    T = [sample(rng, ["TT", "AA", "AT"], Weights(softmax[i, :])) for i in 1:n]
    y = ft(T) + fw(W₁, W₂, W₃) + rand(rng, Normal(0,1), n)

    # Ew[E[Y|t,w]] = ∑ᵤ (ft(T) + fw(w))p(w) = ft(t) + 0.5
    ATE = (ft(case) + 0.5) -  (ft(control) + 0.5)
    return (T = categorical(T),  W₁ = W₁, W₂ = W₂, W₃ = W₃, y = y), ATE
end


function dataset_2_treatments_pb(;rng = StableRNG(123), n=100)
    μY(W₁, W₂, T₁, T₂) = 4T₁ .- 2T₂ .+ 5W₁ .- 3W₂
    W₁ = rand(rng, Normal(), n)
    W₂ = rand(rng, Normal(), n)
    μT₁ = logistic.(0.5W₁ + 1.5W₂)
    T₁ = float(rand(rng, Uniform(), n) .< μT₁)
    μT₂ = logistic.(-1.5W₁ + .5W₂ .+ 1)
    T₂ = float(rand(rng, Uniform(), n) .< μT₂)
    y = μY(W₁, W₂, T₁, T₂) .+ rand(rng, Normal(), n)
    # Those ATEs are MC approximations, only reliable with large samples
    case = ones(n)
    control = zeros(n)
    ATE₁₁₋₀₁ = mean(μY(W₁, W₂, case, case) .- μY(W₁, W₂, control, case))
    ATE₁₁₋₀₀ = mean(μY(W₁, W₂, case, case) .- μY(W₁, W₂, control, control))
    return (
        W₁ = W₁,
        W₂ = W₂,
        T₁ = categorical(T₁),
        T₂ = categorical(T₂),
        y  = y
    ), ATE₁₁₋₀₁, ATE₁₁₋₀₀
end

@testset "Test Double Robustness ATE with two treatment variables" begin
    dataset, ATE₁₁₋₀₁, ATE₁₁₋₀₀ = dataset_2_treatments_pb(;rng = StableRNG(123), n=10_000)
    cache = TMLECache(dataset)
    # Test first ATE, only T₁ treatment varies 
    Ψ = ATE(
        target      = :y,
        treatment   = (T₁=(case=1., control=0.), T₂=(case=1., control=1.)),
        confounders = [:W₁, :W₂]
    )
    # When Q is misspecified but G is well specified
    η_spec = NuisanceSpec(
        MLJModels.DeterministicConstantRegressor(),
        LogisticClassifier(lambda=0)
    )
    tmle_result, cache = tmle!(cache, Ψ, η_spec, verbosity=0)
    test_coverage(tmle_result, ATE₁₁₋₀₁)
    test_fluct_decreases_risk(cache; target_name=:y)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-10)
    test_fluct_mean_inf_curve_lower_than_initial(tmle_result)
    # When Q is well specified but G is misspecified
    η_spec = NuisanceSpec(
        LinearRegressor(),
        ConstantClassifier()
    )
    tmle_result, cache = tmle!(cache, η_spec, verbosity=0)
    test_coverage(tmle_result, ATE₁₁₋₀₁)
    test_fluct_decreases_risk(cache; target_name=:y)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-10)
    test_fluct_mean_inf_curve_lower_than_initial(tmle_result)

    # Test second ATE, two treatment varies 
    Ψ = ATE(
        target      = :y,
        treatment   = (T₁=(case=1., control=0.), T₂=(case=1., control=0.)),
        confounders = [:W₁, :W₂]
    )
    tmle_result, cache = tmle!(cache, Ψ, η_spec, verbosity=0)
    test_coverage(tmle_result, ATE₁₁₋₀₀)
    test_fluct_decreases_risk(cache; target_name=:y)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-10)
    test_fluct_mean_inf_curve_lower_than_initial(tmle_result)
    # When Q is well specified but G is misspecified
    η_spec = NuisanceSpec(
        LinearRegressor(),
        ConstantClassifier()
    )
    tmle_result, cache = tmle!(cache, η_spec, verbosity=0)
    test_coverage(tmle_result, ATE₁₁₋₀₀)
    test_fluct_decreases_risk(cache; target_name=:y)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-10)
    test_fluct_mean_inf_curve_lower_than_initial(tmle_result)
end

@testset "Test Double Robustness ATE on continuous_target_categorical_treatment_pb" begin
    dataset, Ψ₀ = continuous_target_categorical_treatment_pb(;n=10_000, control="TT", case="AA")
    Ψ = ATE(
        target      = :y,
        treatment   = (T=(case="AA", control="TT"),),
        confounders = [:W₁, :W₂, :W₃]
        )
    # When Q is misspecified but G is well specified
    η_spec = NuisanceSpec(
        MLJModels.DeterministicConstantRegressor(),
        LogisticClassifier(lambda=0)
    )
    cache = TMLECache(dataset)
    tmle_result, cache = tmle!(cache, Ψ, η_spec, verbosity=0)
    test_coverage(tmle_result, Ψ₀)
    test_fluct_decreases_risk(cache; target_name=:y)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-10)
    test_fluct_mean_inf_curve_lower_than_initial(tmle_result)
    # The initial estimate is far away
    @test tmle_result.initial == 0
    
    # When Q is well specified but G is misspecified
    η_spec = NuisanceSpec(
        LinearRegressor(),
        ConstantClassifier()
    )

    tmle_result, cache = tmle!(cache, η_spec, verbosity=0)
    test_coverage(tmle_result, Ψ₀)
    test_fluct_decreases_risk(cache, target_name=:y)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-10)
    test_fluct_mean_inf_curve_lower_than_initial(tmle_result)
end

@testset "Test Double Robustness ATE on binary_target_binary_treatment_pb" begin
    dataset, Ψ₀ = binary_target_binary_treatment_pb(;n=10_000)
    Ψ = ATE(
        target = :y,
        treatment = (T=(case=true, control=false),),
        confounders = [:W]
    )
    # When Q is misspecified but G is well specified
    η_spec = NuisanceSpec(
        ConstantClassifier(),
        LogisticClassifier(lambda=0)
    )
    tmle_result, cache = tmle(Ψ, η_spec, dataset, verbosity=0)
    test_coverage(tmle_result, Ψ₀)
    test_fluct_decreases_risk(cache; target_name=:y)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-6)
    test_fluct_mean_inf_curve_lower_than_initial(tmle_result)
    # The initial estimate is far away
    @test tmle_result.initial == 0

    # When Q is well specified but G is misspecified
    η_spec = NuisanceSpec(
        LogisticClassifier(lambda=0),
        ConstantClassifier()
    )
    tmle_result, cache = tmle!(cache, η_spec, verbosity=0)
    test_coverage(tmle_result, Ψ₀)
    test_fluct_decreases_risk(cache; target_name=:y)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-6)
end


@testset "Test Double Robustness ATE on continuous_target_binary_treatment_pb" begin
    dataset, Ψ₀ = continuous_target_binary_treatment_pb(n=10_000)
    Ψ = ATE(
        target      = :y,
        treatment   = (T=(case=true, control=false),),
        confounders = [:W₁, :W₂, :W₃]
    )
    # When Q is misspecified but G is well specified
    η_spec = NuisanceSpec(
        MLJModels.DeterministicConstantRegressor(),
        LogisticClassifier(lambda=0)
    )

    tmle_result, cache = tmle(Ψ, η_spec, dataset, verbosity=0)
    test_coverage(tmle_result, Ψ₀)
    test_fluct_decreases_risk(cache; target_name=:y)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-10)
    test_fluct_mean_inf_curve_lower_than_initial(tmle_result)
    # The initial estimate is far away
    @test tmle_result.initial == 0

    # When Q is well specified but G is misspecified
    η_spec = NuisanceSpec(
        LinearRegressor(),
        ConstantClassifier()
    )
    tmle_result, cache = tmle!(cache, η_spec, verbosity=0)
    test_coverage(tmle_result, Ψ₀)
    test_fluct_decreases_risk(cache; target_name=:y)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-10)
end

end;

true