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
function binary_outcome_binary_treatment_pb(;n=100)
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
    Y = categorical(y)
    dataset = (T=T, W=W, Y=Y)
    # Compute the theoretical ATE
    ATE₁ = py_given_aw(1, 1)*p_w() + (1-p_w())*py_given_aw(1, 0)
    ATE₀ = py_given_aw(0, 1)*p_w() + (1-p_w())*py_given_aw(0, 0)
    ATE = ATE₁ - ATE₀
    # Define the SCM
    scm = StaticConfoundedModel(:Y, :T, :W)
    
    return dataset, scm, ATE
end

"""
From https://www.degruyter.com/document/doi/10.2202/1557-4679.1043/html
The theoretical ATE is 1
"""
function continuous_outcome_binary_treatment_pb(;n=100)
    # Dataset
    rng = StableRNG(123)
    Unif = Uniform(0, 1)
    W = float(rand(rng, Bernoulli(0.5), n, 3))
    W₁, W₂, W₃ = W[:, 1], W[:, 2], W[:, 3]
    t = rand(rng, Unif, n) .< logistic.(0.5W₁ + 1.5W₂ - W₃)
    y = 4t + 25W₁ + 3W₂ - 4W₃ + rand(rng, Normal(0, 0.1), n)
    T = categorical(t)
    dataset = (T = T, W₁ = W₁, W₂ = W₂, W₃ = W₃, Y = y)
    # Theroretical ATE
    ATE = 4
    # Define the SCM
    scm = StaticConfoundedModel(:Y, :T, [:W₁, :W₂, :W₃])

    return dataset, scm, ATE
end

function continuous_outcome_categorical_treatment_pb(;n=100, control="TT", case="AA")
    # Define dataset
    rng = StableRNG(123)
    ft(T) = (T .== "AA") - (T .== "AT") + 2(T .== "TT")
    fw(W₁, W₂, W₃) = 2W₁ + 3W₂ - 4W₃
    W = float(rand(rng, Bernoulli(0.5), n, 3))
    W₁, W₂, W₃ = W[:, 1], W[:, 2], W[:, 3]
    θ = rand(rng, 3, 3)
    softmax = exp.(W*θ) ./ sum(exp.(W*θ), dims=2)
    T = [sample(rng, ["TT", "AA", "AT"], Weights(softmax[i, :])) for i in 1:n]
    y = ft(T) + fw(W₁, W₂, W₃) + rand(rng, Normal(0,1), n)
    dataset = (T = categorical(T),  W₁ = W₁, W₂ = W₂, W₃ = W₃, Y = y)
    # True ATE: Ew[E[Y|t,w]] = ∑ᵤ (ft(T) + fw(w))p(w) = ft(t) + 0.5
    ATE = (ft(case) + 0.5) -  (ft(control) + 0.5)
    # Define the SCM
    scm = StaticConfoundedModel(:Y, :T, [:W₁, :W₂, :W₃])
    return dataset, scm, ATE
end


function dataset_2_treatments_pb(;rng = StableRNG(123), n=100)
    # Dataset
    μY(W₁, W₂, T₁, T₂) = 4T₁ .- 2T₂ .+ 5W₁ .- 3W₂
    W₁ = rand(rng, Normal(), n)
    W₂ = rand(rng, Normal(), n)
    μT₁ = logistic.(0.5W₁ + 1.5W₂)
    T₁ = float(rand(rng, Uniform(), n) .< μT₁)
    μT₂ = logistic.(-1.5W₁ + .5W₂ .+ 1)
    T₂ = float(rand(rng, Uniform(), n) .< μT₂)
    y = μY(W₁, W₂, T₁, T₂) .+ rand(rng, Normal(), n)
    dataset = (
        W₁ = W₁,
        W₂ = W₂,
        T₁ = categorical(T₁),
        T₂ = categorical(T₂),
        Y  = y
    )
    # Those ATEs are MC approximations, only reliable with large samples
    case = ones(n)
    control = zeros(n)
    ATE₁₁₋₀₁ = mean(μY(W₁, W₂, case, case) .- μY(W₁, W₂, control, case))
    ATE₁₁₋₀₀ = mean(μY(W₁, W₂, case, case) .- μY(W₁, W₂, control, control))
    # Define the SCM
    scm = StaticConfoundedModel([:Y], [:T₁, :T₂], [:W₁, :W₂])

    return dataset, scm , (ATE₁₁₋₀₁, ATE₁₁₋₀₀)
end

@testset "Test Double Robustness ATE on continuous_outcome_categorical_treatment_pb" begin
    dataset, scm, Ψ₀ = continuous_outcome_categorical_treatment_pb(;n=10_000, control="TT", case="AA")
    Ψ = ATE(
        scm       = scm,
        outcome   = :Y,
        treatment = (T=(case="AA", control="TT"),),
        )
    # When Q is misspecified but G is well specified
    scm.Y.model = TreatmentTransformer() |> MLJModels.DeterministicConstantRegressor()
    scm.T.model = LogisticClassifier(lambda=0)

    tmle_result, fluctuation_mach = tmle(Ψ, dataset, verbosity=0)
    test_coverage(tmle_result, Ψ₀)
    test_fluct_decreases_risk(Ψ, fluctuation_mach)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-10)
    test_fluct_mean_inf_curve_lower_than_initial(tmle_result)
    # The initial estimate is far away
    @test tmle_result.initial == 0
    
    # When Q is well specified but G is misspecified
    scm.Y.model = TreatmentTransformer() |> LinearRegressor()
    scm.T.model = ConstantClassifier()

    tmle_result, fluctuation_mach = tmle(Ψ, dataset, verbosity=0)
    test_coverage(tmle_result, Ψ₀)
    test_fluct_decreases_risk(Ψ, fluctuation_mach)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-10)
end

@testset "Test Double Robustness ATE on binary_outcome_binary_treatment_pb" begin
    dataset, scm, Ψ₀ = binary_outcome_binary_treatment_pb(;n=10_000)
    Ψ = ATE(
        scm = scm,
        outcome = :Y,
        treatment = (T=(case=true, control=false),),
    )
    # When Q is misspecified but G is well specified
    scm.Y.model = TreatmentTransformer() |> ConstantClassifier()
    scm.T.model = LogisticClassifier(lambda=0)

    tmle_result, fluctuation_mach = tmle(Ψ, dataset, verbosity=0)
    test_coverage(tmle_result, Ψ₀)
    test_fluct_decreases_risk(Ψ, fluctuation_mach)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-6)
    test_fluct_mean_inf_curve_lower_than_initial(tmle_result)
    # The initial estimate is far away
    @test tmle_result.initial == 0

    # When Q is well specified but G is misspecified
    scm.Y.model = TreatmentTransformer() |> LogisticClassifier(lambda=0)
    scm.T.model = ConstantClassifier()

    tmle_result, fluctuation_mach = tmle(Ψ, dataset, verbosity=0)
    test_coverage(tmle_result, Ψ₀)
    test_fluct_decreases_risk(Ψ, fluctuation_mach)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-6)
end


@testset "Test Double Robustness ATE on continuous_outcome_binary_treatment_pb" begin
    dataset, scm, Ψ₀ = continuous_outcome_binary_treatment_pb(n=10_000)
    Ψ = ATE(
        scm=scm,
        outcome      = :Y,
        treatment   = (T=(case=true, control=false),),
    )
    # When Q is misspecified but G is well specified
    scm.Y.model = TreatmentTransformer() |> MLJModels.DeterministicConstantRegressor()
    scm.T.model = LogisticClassifier(lambda=0)

    tmle_result, fluctuation_mach = tmle(Ψ, dataset, verbosity=0)
    test_coverage(tmle_result, Ψ₀)
    test_fluct_decreases_risk(Ψ, fluctuation_mach)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-10)
    test_fluct_mean_inf_curve_lower_than_initial(tmle_result)
    # The initial estimate is far away
    @test tmle_result.initial == 0

    # When Q is well specified but G is misspecified
    scm.Y.model = TreatmentTransformer() |> LinearRegressor()
    scm.T.model = ConstantClassifier()

    tmle_result, fluctuation_mach = tmle(Ψ, dataset, verbosity=0)
    test_coverage(tmle_result, Ψ₀)
    test_fluct_decreases_risk(Ψ, fluctuation_mach)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-10)
end

@testset "Test Double Robustness ATE with two treatment variables" begin
    dataset, scm, (ATE₁₁₋₀₁, ATE₁₁₋₀₀) = dataset_2_treatments_pb(;rng = StableRNG(123), n=50_000)
    # Test first ATE, only T₁ treatment varies 
    Ψ = ATE(
        scm=scm,
        outcome      = :Y,
        treatment   = (T₁=(case=1., control=0.), T₂=(case=1., control=1.)),
    )
    # When Q is misspecified but G is well specified
    scm.Y.model = TreatmentTransformer() |> MLJModels.DeterministicConstantRegressor()
    scm.T₁.model = LogisticClassifier(lambda=0)
    scm.T₂.model = LogisticClassifier(lambda=0)

    tmle_result, fluctuation_mach = tmle(Ψ, dataset, verbosity=0)
    test_coverage(tmle_result, ATE₁₁₋₀₁)
    test_fluct_decreases_risk(Ψ, fluctuation_mach)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-10)
    test_fluct_mean_inf_curve_lower_than_initial(tmle_result)
    # When Q is well specified but G is misspecified
    scm.Y.model = TreatmentTransformer() |> LinearRegressor()
    scm.T₁.model = ConstantClassifier()
    scm.T₂.model = ConstantClassifier()

    tmle_result, fluctuation_mach = tmle(Ψ, dataset, verbosity=0)
    test_coverage(tmle_result, ATE₁₁₋₀₁)
    test_fluct_decreases_risk(Ψ, fluctuation_mach)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-10)

    # Test second ATE, two treatment varies 
    Ψ = ATE(
        scm = scm,
        outcome      = :Y,
        treatment   = (T₁=(case=1., control=0.), T₂=(case=1., control=0.)),
    )
    tmle_result, fluctuation_mach = tmle(Ψ, dataset, verbosity=0)
    test_coverage(tmle_result, ATE₁₁₋₀₀)
    test_fluct_decreases_risk(Ψ, fluctuation_mach)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-10)
    test_fluct_mean_inf_curve_lower_than_initial(tmle_result)

    # When Q is well specified but G is misspecified
    scm.Y.model =  TreatmentTransformer() |> MLJModels.DeterministicConstantRegressor()
    scm.T₁.model = LogisticClassifier(lambda=0)
    scm.T₂.model = LogisticClassifier(lambda=0)

    tmle_result, fluctuation_mach = tmle(Ψ, dataset, verbosity=0)
    test_coverage(tmle_result, ATE₁₁₋₀₀)
    test_fluct_decreases_risk(Ψ, fluctuation_mach)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-10)
end

end;

true