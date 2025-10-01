#!/usr/bin/env julia

"""
Example: LASSO Collaborative TMLE

Demonstrates CV-based variable selection in high-dimensional causal inference.
"""

using Pkg
Pkg.activate(".")

using TMLE
using Random
using DataFrames
using CategoricalArrays
using GLMNet
using Statistics
using Distributions
using LinearAlgebra
using StatsBase  # For sample() function

println("🧬 LASSO Collaborative TMLE Example")
println("=" ^ 50)

Random.seed!(123)

function sim3(; n=1000, p=100, rho=0.9, k=20, amplitude=1.0, amplitude2=1.0, k2=20)
    """
    Generate high-dimensional data with correlated confounders
    
    Parameters:
    - n: sample size  
    - p: number of confounders
    - rho: correlation parameter for Toeplitz covariance
    - k: number of non-zero coefficients for outcome model
    - amplitude: amplitude for outcome coefficients
    - amplitude2: amplitude for propensity score coefficients  
    - k2: number of non-zero coefficients for propensity score
    """
    
    function toeplitz_cov(p, rho)
        return [rho^abs(i-j) for i in 1:p, j in 1:p]
    end
    
    Sigma = toeplitz_cov(p, rho)
    mv_normal = MvNormal(zeros(p), Sigma)
    W_raw = rand(mv_normal, n)'  
    W = (W_raw .- mean(W_raw, dims=1)) ./ std(W_raw, dims=1)
    
    nonzero2 = sample(1:p, k2, replace=false)
    signs2 = sample([-1, 1], p, replace=true)
    beta = amplitude2 * signs2 .* [i in nonzero2 for i in 1:p]
    
    logit_p = W * beta
    prob_A = 1 ./ (1 .+ exp.(-logit_p))
    A = rand.(Bernoulli.(prob_A))
    
    nonzero = sample(1:p, k, replace=false)
    signs = sample([-1, 1], p, replace=true)
    gamma = amplitude * signs .* [i in nonzero for i in 1:p]
    
    Y = 2.0 * A + W * gamma + randn(n)
    
    W_df = DataFrame(W, [Symbol("W$i") for i in 1:p])
    data = hcat(W_df, DataFrame(A=categorical(A), Y=Y))
    
    return data, nonzero, nonzero2
end

println("\n📊 Generating high-dimensional simulation data...")
n = 10000
p = 50
rho = 0.5

dataset, true_outcome_vars, true_ps_vars = sim3(n=n, p=p, rho=rho, k=20, k2=20)

all_confounders = [Symbol("W$i") for i in 1:p]

println("Generated dataset: $n observations, $p confounders")
println("True treatment effect: 2.0")
println("Treatment prevalence: $(round(mean(dataset.A .== 1), digits=3))")

estimand = ATE(
    outcome = :Y,
    treatment_values = (A = (case = 1, control = 0),),
    treatment_confounders = (A = all_confounders,)
)

println("\n🔬 CAUSAL INFERENCE COMPARISON")
println("=" ^ 50)

# Standard TMLE
println("\n1️⃣ Standard TMLE (uses all $p confounders)")
standard_estimator = Tmle()
standard_result, _ = standard_estimator(estimand, dataset; verbosity=0)
std_estimate = estimate(standard_result)
println("   Estimate: $(round(std_estimate, digits=3))")

# LASSO CTMLE with cv lambda selection
println("\n2️⃣ LASSO CTMLE (cv lambda selection)")
lasso_strategy = LassoCTMLE(
    confounders = all_confounders,
    patience = 6,
    alpha = 1.0
)

lasso_estimator = Tmle(collaborative_strategy = lasso_strategy)
lasso_result, _ = lasso_estimator(estimand, dataset; verbosity=1)
lasso_estimate = estimate(lasso_result)
println("   Estimate: $(round(lasso_estimate, digits=3))")

println("\n📊 RESULTS SUMMARY")
println("=" ^ 50)
println("True treatment effect:    2.000")
println("Standard TMLE:           $(round(std_estimate, digits=3))")
println("LASSO CTMLE:             $(round(lasso_estimate, digits=3))")

println("\nAbsolute deviations from truth:")
println("Standard TMLE:           $(round(abs(std_estimate - 2.0), digits=3))")
println("LASSO CTMLE:             $(round(abs(lasso_estimate - 2.0), digits=3))")

println("\n✅ Example completed successfully!")
