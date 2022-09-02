module TestWarmRestart

using Test
using StableRNGs
using TMLE
using DataFrames
using Distributions
using MLJLinearModels
using CategoricalArrays

# A) End to end tests to perform:

# Parameters:
# Probably on simple problems, and test for 
# 1/ coverage, ie correctness
# 2/ comparison with initial estimate

# - Mean 1 treatment
# - Mean multiple treatments

# - ATE 1 treatment: check
# - ATE multiple treatments

# - IATE 2 treatments
# - IATE 3 treatments

# Updates:
# - η_spec: G
# - η_spec: Q
# - parameter: new treatment
# - parameter: new confounder
# - parameter: new covariate
# - parameter: new target
# - parameter: new parameter

# B) Integration test with various Tables
# - DataFrames
# - Arrow
# - NamedTuple


# C) Composition of EstimationResult

function covers(result, Ψ₀; level=0.05)
    test = OneSampleTTest(result, Ψ₀)
    pval = level ≤ pvalue(test)
    lower, upper = confint(test)
    covered = lower ≤ Ψ₀ ≤ upper
    return pval && covered
end

closer_than_initial(tmle_result, initial_result, Ψ₀) =
    abs(TMLE.estimate(tmle_result) - Ψ₀) ≤ abs(TMLE.estimate(initial_result) - Ψ₀)

"""
Results derived by hand for this dataset:

- ATE(y, T₁, W₁, W₂, C₁) = 1 - 4E[C₁] = 1 - 2 = -1
- ATE(y, T₂, W₁, W₂, C₁) = E[W₂] = 0.5
"""
function build_dataset(;n=100)
    rng = StableRNG(123)
    # Confounders
    W₁ = rand(rng, Uniform(), n)
    W₂ = rand(rng, Uniform(), n)
    # Covariates
    C₁ = rand(rng, n)
    # Treatment | Confounders
    T₁ = rand(rng, Uniform(), n) .< TMLE.expit(0.5sin.(W₁) .- 1.5W₂)
    T₂ = rand(rng, Uniform(), n) .< TMLE.expit(-3W₁ - 1.5W₂)
    # target | Confounders, Covariates, Treatments - T₁.*T₂
    y₁ = 1 .+ 2W₁ .+ 3W₂ .- 4C₁.*T₁ .+ T₁ + T₂.*W₂ .+ rand(rng, Normal(0, 0.1), n)
    y₂ = 1 .+ 10T₂ .+ rand(rng, Normal(0, 0.1), n)
    return DataFrame(
        T₁ = categorical(T₁),
        T₂ = categorical(T₂),
        W₁ = W₁, 
        W₂ = W₂,
        C₁ = C₁,
        y₁ = y₁,
        y₂ = y₂
        )
end

@testset "Test Warm restart for ATE estimation" begin
    dataset = build_dataset(;n=1000)
    # Define the parameter of interest
    Ψ = ATE(
        target=:y₁,
        treatment=(T₁=(case=1, control=0),),
        confounders=[:W₁, :W₂],
        covariates=[:C₁]
    )
    # Define the nuisance parameters specification
    η_spec = (
        Q = LinearRegressor(),
        G = LogisticClassifier(lambda=0)
    )
    # Run TMLE
    log_sequence = (
        (:info, "Fitting the nuisance parameters..."),
        (:info, "→ Fitting P(T|W)"),
        (:info, "→ Fitting Encoder"),
        (:info, "→ Fitting E[Y|X]"),
        (:info, "Targeting the nuisance parameters..."),
        (:info, "Thank you.")
    )
    tmle_result, initial_result, cache = @test_logs log_sequence... tmle(Ψ, η_spec, dataset; verbosity=1);
    # The TMLE covers the ground truth but the initial estimate does not
    Ψ₀ = -1
    @test covers(tmle_result, Ψ₀)
    @test closer_than_initial(tmle_result, initial_result, Ψ₀)

    # Update the treatment specification
    # Nuisance parameters should not fitted again
    Ψ = ATE(
        target=:y₁,
        treatment=(T₁=(case=0, control=1),),
        confounders=[:W₁, :W₂],
        covariates=[:C₁]
    )
    log_sequence = (
        (:info, "Fitting the nuisance parameters..."),
        (:info, "→ Reusing previous P(T|W)"),
        (:info, "→ Reusing previous Encoder"),
        (:info, "→ Reusing previous E[Y|X]"),
        (:info, "Targeting the nuisance parameters..."),
        (:info, "Thank you.")
    )
    tmle_result, initial_result, cache = @test_logs log_sequence... tmle!(cache, Ψ, verbosity=1);
    Ψ₀ = 1
    @test covers(tmle_result, Ψ₀)
    @test closer_than_initial(tmle_result, initial_result, Ψ₀)

    # Remove the covariate variable, this will trigger the refit of Q
    Ψ = ATE(
        target=:y₁,
        treatment=(T₁=(case=1, control=0),),
        confounders=[:W₁, :W₂],
    )
    log_sequence = (
        (:info, "Fitting the nuisance parameters..."),
        (:info, "→ Reusing previous P(T|W)"),
        (:info, "→ Reusing previous Encoder"),
        (:info, "→ Fitting E[Y|X]"),
        (:info, "Targeting the nuisance parameters..."),
        (:info, "Thank you.")
    )
    tmle_result, initial_result, cache = @test_logs log_sequence... tmle!(cache, Ψ, verbosity=1);
    Ψ₀ = -1
    @test covers(tmle_result, Ψ₀)
    @test closer_than_initial(tmle_result, initial_result, Ψ₀)

    # Change the treatment
    # This will trigger the refit of all η
    Ψ = ATE(
        target=:y₁,
        treatment=(T₂=(case=1, control=0),),
        confounders=[:W₁, :W₂],
    )
    log_sequence = (
        (:info, "Fitting the nuisance parameters..."),
        (:info, "→ Fitting P(T|W)"),
        (:info, "→ Fitting Encoder"),
        (:info, "→ Fitting E[Y|X]"),
        (:info, "Targeting the nuisance parameters..."),
        (:info, "Thank you.")
    )
    tmle_result, initial_result, cache = @test_logs log_sequence... tmle!(cache, Ψ, verbosity=1);
    Ψ₀ = 0.5
    @test covers(tmle_result, Ψ₀)
    @test closer_than_initial(tmle_result, initial_result, Ψ₀)

    # Remove a confounding variable
    # This will trigger the refit of Q and G
    # Since we are not accounting for all confounders 
    # we can't have ground truth coverage on this setting
    Ψ = ATE(
        target=:y₁,
        treatment=(T₂=(case=1, control=0),),
        confounders=[:W₁],
    )
    log_sequence = (
        (:info, "Fitting the nuisance parameters..."),
        (:info, "→ Fitting P(T|W)"),
        (:info, "→ Reusing previous Encoder"),
        (:info, "→ Fitting E[Y|X]"),
        (:info, "Targeting the nuisance parameters..."),
        (:info, "Thank you.")
    )
    tmle_result, initial_result, cache = @test_logs log_sequence... tmle!(cache, Ψ, verbosity=1);
    
    # Change the target
    # This will trigger the refit of Q only
    Ψ = ATE(
        target=:y₂,
        treatment=(T₂=(case=1, control=0),),
        confounders=[:W₁],
    )
    log_sequence = (
        (:info, "Fitting the nuisance parameters..."),
        (:info, "→ Reusing previous P(T|W)"),
        (:info, "→ Reusing previous Encoder"),
        (:info, "→ Fitting E[Y|X]"),
        (:info, "Targeting the nuisance parameters..."),
        (:info, "Thank you.")
    )
    tmle_result, initial_result, cache = @test_logs log_sequence... tmle!(cache, Ψ, verbosity=1);
    Ψ₀ = 10
    @test covers(tmle_result, Ψ₀)
    @test closer_than_initial(tmle_result, initial_result, Ψ₀)

end

end
