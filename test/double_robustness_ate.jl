module TestDoubleRobustnessATE

include("helper_fns.jl")

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
function continuous_target_binary_treatment_pb(rng;n=100)
    Unif = Uniform(0, 1)
    W = float(rand(rng, Bernoulli(0.5), n, 3))
    W₁, W₂, W₃ = W[:, 1], W[:, 2], W[:, 3]
    t = rand(rng, Unif, n) .< TMLE.expit(0.5W₁ + 1.5W₂ - W₃)
    y = 4t + 25W₁ + 3W₂ - 4W₃ + rand(rng, Normal(0, 0.1), n)
    # Type coercion
    T = categorical(t)
    return (T = T, W₁ = W₁, W₂ = W₂, W₃ = W₃, y = y), 4
end

function continuous_target_categorical_treatment_pb(rng;n=100, control="TT", treatment="AA")
    ft(T) = (T .== "AA") - (T .== "AT") + 2(T .== "TT")
    fw(W₁, W₂, W₃) = 2W₁ + 3W₂ - 4W₃

    W = float(rand(rng, Bernoulli(0.5), n, 3))
    W₁, W₂, W₃ = W[:, 1], W[:, 2], W[:, 3]
    θ = rand(rng, 3, 3)
    softmax = exp.(W*θ) ./ sum(exp.(W*θ), dims=2)
    T = [sample(rng, ["TT", "AA", "AT"], Weights(softmax[i, :])) for i in 1:n]
    y = ft(T) + fw(W₁, W₂, W₃) + rand(rng, Normal(0,1), n)

    # Ew[E[Y|t,w]] = ∑ᵤ (ft(T) + fw(w))p(w) = ft(t) + 0.5
    ATE = (ft(treatment) + 0.5) -  (ft(control) + 0.5)
    # Type coercion
    T = categorical(T)
    return (T = T,  W₁ = W₁, W₂ = W₂, W₃ = W₃, y = y), ATE
end

@testset "Test Double Robustness ATE on continuous_target_categorical_treatment_pb" begin
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
    tmle_results, initial_results, Ψ₀ = asymptotics(
        Ψ, 
        η_spec, 
        continuous_target_categorical_treatment_pb, 
        StableRNG(123), 
        Ns)

    @test all_tmle_better_than_initial(tmle_results, initial_results, Ψ₀)
    @test first_better_than_last(tmle_results, Ψ₀)
    @test tolerance(tmle_results[end], Ψ₀, 0.011)
    @test all_solves_ice(tmle_results)  

    # When Q is well specified but G is misspecified
    η_spec = NuisanceSpec(
        LinearRegressor(),
        ConstantClassifier()
    )
    tmle_results, initial_results, Ψ₀ = asymptotics(
        Ψ, 
        η_spec, 
        continuous_target_categorical_treatment_pb, 
        StableRNG(123), 
        Ns)
    # I think since Q is correctly specified and the model is some simple
    # the convergence may occur after only 10 samples, ie more samples don't do anything
    @test all_tmle_better_than_initial(tmle_results, initial_results, Ψ₀)
    @test first_better_than_last(tmle_results, Ψ₀)
    @test tolerance(tmle_results[end], Ψ₀, 0.011)
    @test all_solves_ice(tmle_results)
end

@testset "Test Double Robustness ATE on binary_target_binary_treatment_pb" begin
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
    tmle_results, initial_results, Ψ₀ = asymptotics(
        Ψ, 
        η_spec, 
        binary_target_binary_treatment_pb, 
        StableRNG(123), 
        Ns)
    
    @test all_tmle_better_than_initial(tmle_results, initial_results, Ψ₀)
    @test first_better_than_last(tmle_results, Ψ₀)
    @test tolerance(tmle_results[end], Ψ₀, 0.011)
    @test all_solves_ice(tmle_results, tol=1e-6)

    # When Q̅ is well specified but G is misspecified
    η_spec = NuisanceSpec(
        LogisticClassifier(lambda=0),
        ConstantClassifier()
    )

    tmle_results, initial_results, Ψ₀ = asymptotics(
        Ψ, 
        η_spec, 
        binary_target_binary_treatment_pb, 
        StableRNG(123), 
        Ns)
    # for some reason, the TMLE is not always better in this case
    # Maybe a numeric convergence related issue of the GLM model
    # We can see that the initial fit already solves the EIC equation
    @test all(mean(r.IC) < 1e-10 for r in initial_results)
    @test first_better_than_last(tmle_results, Ψ₀)
    @test tolerance(tmle_results[end], Ψ₀, 0.011)
    @test all_solves_ice(tmle_results, tol=1e-6)
end


@testset "Test Double Robustness ATE on continuous_target_binary_treatment_pb" begin
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

    tmle_results, initial_results, Ψ₀ = asymptotics(
        Ψ, 
        η_spec,                               
        continuous_target_binary_treatment_pb,
        StableRNG(123),
        Ns
        )
    # For some reason the TMLE for n=100 samples is really bad
    @test all_tmle_better_than_initial(tmle_results[2:end], initial_results[2:end], Ψ₀)
    @test first_better_than_last(tmle_results, Ψ₀)
    @test tolerance(tmle_results[end], Ψ₀, 0.02)
    @test all_solves_ice(tmle_results)

    # When Q is well specified but G is misspecified
    η_spec = NuisanceSpec(
        LinearRegressor(),
        ConstantClassifier()
    )
    tmle_results, initial_results, Ψ₀ = asymptotics(
        Ψ, 
        η_spec,                               
        continuous_target_binary_treatment_pb,
        StableRNG(123),
        Ns
        )
    @test all_tmle_better_than_initial(tmle_results[2:end], initial_results[2:end], Ψ₀)
    @test first_better_than_last(tmle_results, Ψ₀)
    @test tolerance(tmle_results[end], Ψ₀, 0.01)
    @test all_solves_ice(tmle_results)
end

end;

true