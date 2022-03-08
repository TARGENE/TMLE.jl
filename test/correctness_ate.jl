module TestDoubleRobustnessATE

include("helper_fns.jl")

using TMLE
using Random
using Test
using Distributions
using MLJ
using StableRNGs
using StatsBase
using HypothesisTests

LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels verbosity=0
LinearRegressor = @load LinearRegressor pkg=MLJLinearModels verbosity = 0

Ns = [100, 1000, 10000, 100000]

function binary_target_binary_treatment_pb(rng;n=100)
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
    W = (W=convert(Array{Float64}, w),)
    t = (t=categorical(t),)
    y = categorical(y)
    # Compute the theoretical ATE
    ATE₁ = py_given_aw(1, 1)*p_w() + (1-p_w())*py_given_aw(1, 0)
    ATE₀ = py_given_aw(0, 1)*p_w() + (1-p_w())*py_given_aw(0, 0)
    ATE = ATE₁ - ATE₀
    
    return t, W, y, ATE
end

"""
From https://www.degruyter.com/document/doi/10.2202/1557-4679.1043/html
The theoretical ATE is 1
"""
function continuous_target_binary_treatment_pb(rng;n=100)
    Unif = Uniform(0, 1)
    W = float(rand(rng, Bernoulli(0.5), n, 3))
    t = rand(rng, Unif, n) .< TMLE.expit(0.5W[:, 1] + 1.5W[:, 2] - W[:,3])
    y = t + 2W[:, 1] + 3W[:, 2] - 4W[:, 3] + rand(rng, Normal(0,1), n)
    # Type coercion
    W = MLJ.table(W)
    t = (t=categorical(t),)
    return t, W, y, 1
end


function continuous_target_categorical_treatment_pb(rng;n=100, control="TT", treatment="AA")
    ft(T) = (T .== "AA") - (T .== "AT") + 2(T .== "TT")
    fw(W) = 2W[:, 1] + 3W[:, 2] - 4W[:, 3]

    W = float(rand(rng, Bernoulli(0.5), n, 3))
    θ = rand(rng, 3, 3)
    softmax = exp.(W*θ) ./ sum(exp.(W*θ), dims=2)
    T = [sample(rng, ["TT", "AA", "AT"], Weights(softmax[i, :])) for i in 1:n]
    y = ft(T) + fw(W) + rand(rng, Normal(0,1), n)

    # Ew[E[Y|t,w]] = ∑ᵤ (ft(T) + fw(w))p(w) = ft(t) + 0.5
    ATE = (ft(treatment) + 0.5) -  (ft(control) + 0.5)
    # Type coercion
    W = MLJ.table(W)
    T = (T=categorical(T),)
    return T, W, y, ATE
end


@testset "Test Double Robustness ATE on continuous_target_categorical_treatment_pb" begin
    query = Query((T="AA",), (T="TT",))
    # When Q̅ is misspecified but G is well specified
    Q̅ = MLJ.DeterministicConstantRegressor()
    G = LogisticClassifier()
    tmle = TMLEstimator(Q̅, G, query)

    abs_mean_rel_errors, abs_vars = asymptotics(
            tmle,                                 
            continuous_target_categorical_treatment_pb,
            StableRNG(123),
            Ns
            )
    @test all(abs_mean_rel_errors .< [26, 9, 3.5, 0.6])
    @test all(abs_vars .< [0.09, 0.02, 0.002, 3.9e-5])

    # When Q̅ is well specified but G is misspecified
    Q̅ = LinearRegressor()
    G = ConstantClassifier()
    tmle = TMLEstimator(Q̅, G, query)

    abs_mean_rel_errors, abs_vars = asymptotics(
            tmle,                                 
            continuous_target_categorical_treatment_pb,
            StableRNG(123),
            Ns
            )
    @test all(abs_mean_rel_errors .< [12, 9.1, 3.4, 0.5])
    @test all(abs_vars .< [0.03, 0.02, 0.002, 3.5e-5])
end

@testset "Test Double Robustness ATE on binary_target_binary_treatment_pb" begin
    query = Query((t=true,), (t=false,))
    # When Q̅ is misspecified but G is well specified
    Q̅ = ConstantClassifier()
    G = LogisticClassifier()
    tmle = TMLEstimator(Q̅, G, query)

    abs_mean_rel_errors, abs_vars = asymptotics(
            tmle,                                 
            binary_target_binary_treatment_pb,
            StableRNG(123),
            Ns
            )
    @test all(abs_mean_rel_errors .< [9.2, 4.1, 1.6, 0.5])
    @test all(abs_vars .< [0.004, 0.001, 4.7e-5, 7.8e-6])

    # When Q̅ is well specified but G is misspecified
    Q̅ = LogisticClassifier()
    G = ConstantClassifier()
    tmle = TMLEstimator(Q̅, G, query)

    abs_mean_rel_errors, abs_vars = asymptotics(
            tmle,                                 
            binary_target_binary_treatment_pb,
            StableRNG(123),
            Ns
            )
    @test all(abs_mean_rel_errors .< [9.2, 4.1, 1.2, 0.4])
    @test all(abs_vars .< [0.004, 0.001, 4.9e-5, 8.2e-6])
end


@testset "Test Double Robustness ATE on continuous_target_binary_treatment_pb" begin
    query = Query((t=true,), (t=false,))
    # When Q̅ is misspecified but G is well specified
    Q̅ = MLJ.DeterministicConstantRegressor()
    G = LogisticClassifier()
    tmle = TMLEstimator(Q̅, G, query)

    abs_mean_rel_errors, abs_vars = asymptotics(
            tmle,                                 
            continuous_target_binary_treatment_pb,
            StableRNG(123),
            Ns
            )
    @test all(abs_mean_rel_errors .< [67, 8.1, 2.8, 1.2])
    @test all(abs_vars .< [0.4, 0.01, 0.002, 0.0002])

    # When Q̅ is well specified but G is misspecified
    Q̅ = LinearRegressor()
    G = ConstantClassifier()
    tmle = TMLEstimator(Q̅, G, query)

    abs_mean_rel_errors, abs_vars = asymptotics(
            tmle,                                 
            continuous_target_binary_treatment_pb,
            StableRNG(123),
            Ns
            )
    @test all(abs_mean_rel_errors .< [14.8, 5.4, 2.3, 0.9])
    @test all(abs_vars .< [0.04, 0.004, 0.0008, 0.0002])
end

@testset "Test multi-queries/multi-targets" begin
    rng = StableRNG(123)
    n = 100
    queries = (
        Query((T="AT",), (T="TT",)),
        Query((T="AA",), (T="AT",))
    )

    Q̅ = LinearRegressor()
    G = LogisticClassifier()
    tmle = TMLEstimator(Q̅, G, queries...)

    t, W, y₁, ATE₁ = continuous_target_categorical_treatment_pb(rng;n=n, control="TT", treatment="AT")
    _, _, _, ATE₂ = continuous_target_categorical_treatment_pb(rng; control="AT", treatment="AA")
    y₂ = rand(rng, n)

    y = (y₁=y₁, y₂=y₂)
    mach = machine(tmle, t, W, y)
    fit!(mach, verbosity=0)

    # Check results for the first target
    # First query: ATE₁
    conf_interval = confint(ztest(mach, 1, 1))
    @test conf_interval[1] <= ATE₁ <= conf_interval[2]
    # Second query: ATE₂
    conf_interval = confint(ztest(mach, 1, 2))
    @test conf_interval[1] <= ATE₂ <= conf_interval[2]

    # Check results for the second target which is just Random
    conf_interval = confint(ztest(mach, 2, 1))
    # First query:
    @test conf_interval[1] <= 0 <= conf_interval[2]
    # Second query:  
    conf_interval = confint(ztest(mach, 2, 2))
    @test conf_interval[1] <= 0 <= conf_interval[2]
end


end;

true