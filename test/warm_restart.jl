module TestWarmRestart

using Test
using Tables
using StableRNGs
using TMLE
using DataFrames
using Distributions
using MLJLinearModels
using CategoricalArrays
using LogExpFunctions

function covers(result, Ψ₀; level=0.05)
    test = OneSampleTTest(result, Ψ₀)
    pval = level ≤ pvalue(test)
    lower, upper = confint(test)
    covered = lower ≤ Ψ₀ ≤ upper
    return pval && covered
end

closer_than_initial(tmle_result, Ψ₀) =
    abs(TMLE.estimate(tmle_result) - Ψ₀) ≤ abs(initial_estimate(tmle_result) - Ψ₀)

"""
Results derived by hand for this dataset:

- ATE(y₁, T₁, W₁, W₂, C₁) = 1 - 4E[C₁] = 1 - 2 = -1
- ATE(y₁, T₂, W₁, W₂, C₁) = E[W₂] = 0.5
- ATE(y₁, (T₁, T₂), W₁, W₂, C₁) = E[-4C₁ + 1 + W₂] = -0.5
- ATE(y₂, T₂, W₁, W₂, C₁) = 10
"""
function build_dataset(;n=100)
    rng = StableRNG(123)
    # Confounders
    W₁ = rand(rng, Uniform(), n)
    W₂ = rand(rng, Uniform(), n)
    # Covariates
    C₁ = rand(rng, n)
    # Treatment | Confounders
    T₁ = rand(rng, Uniform(), n) .< logistic.(0.5sin.(W₁) .- 1.5W₂)
    T₂ = rand(rng, Uniform(), n) .< logistic.(-3W₁ - 1.5W₂)
    # target | Confounders, Covariates, Treatments
    y₁ = 1 .+ 2W₁ .+ 3W₂ .- 4C₁.*T₁ .+ T₁ + T₂.*W₂ .+ rand(rng, Normal(0, 0.1), n)
    y₂ = 1 .+ 10T₂ .+ rand(rng, Normal(0, 0.1), n)
    y₃ = 1 .+ 2W₁ .+ 3W₂ .- 4C₁.*T₁ .- 2T₂.*T₁.*W₂ .+ rand(rng, Normal(0, 0.1), n)
    return (
        T₁ = categorical(T₁),
        T₂ = categorical(T₂),
        W₁ = W₁, 
        W₂ = W₂,
        C₁ = C₁,
        y₁ = y₁,
        y₂ = y₂,
        y₃ = y₃
        )
end

table_types = (Tables.columntable, DataFrame)

@testset "Test Warm restart: ATE single treatment, $tt" for tt in table_types
    dataset = tt(build_dataset(;n=1000))
    # Define the parameter of interest
    Ψ = ATE(
        target=:y₁,
        treatment=(T₁=(case=true, control=false),),
        confounders=[:W₁, :W₂],
        covariates=[:C₁]
    )
    # Define the nuisance parameters specification
    η_spec = NuisanceSpec(
        LinearRegressor(),
        LogisticClassifier(lambda=0)
    )
    # Run TMLE
    log_sequence = (
        (:info, "Fitting the nuisance parameters..."),
        (:info, "→ Fitting P(T|W)"),
        (:info, "→ Fitting Encoder"),
        (:info, "→ Fitting E[Y|X]"),
        (:info, "Targeting the nuisance parameters..."),
        (:info, "Done.")
    )
    tmle_result, cache = @test_logs log_sequence... tmle(Ψ, η_spec, dataset; verbosity=1);
    # The TMLE covers the ground truth but the initial estimate does not
    Ψ₀ = -1
    @test covers(tmle_result, Ψ₀)
    #@test closer_than_initial(tmle_result, initial_result, Ψ₀)

    # Update the treatment specification
    # Nuisance parameters should not fitted again
    Ψ = ATE(
        target=:y₁,
        treatment=(T₁=(case=false, control=true),),
        confounders=[:W₁, :W₂],
        covariates=[:C₁]
    )
    log_sequence = (
        (:info, "Fitting the nuisance parameters..."),
        (:info, "→ Reusing previous P(T|W)"),
        (:info, "→ Reusing previous Encoder"),
        (:info, "→ Reusing previous E[Y|X]"),
        (:info, "Targeting the nuisance parameters..."),
        (:info, "Done.")
    )
    tmle_result, cache = @test_logs log_sequence... tmle!(cache, Ψ, verbosity=1);
    Ψ₀ = 1
    @test covers(tmle_result, Ψ₀)
    #@test closer_than_initial(tmle_result, initial_result, Ψ₀)

    # Remove the covariate variable, this will trigger the refit of Q
    Ψ = ATE(
        target=:y₁,
        treatment=(T₁=(case=true, control=false),),
        confounders=[:W₁, :W₂],
    )
    log_sequence = (
        (:info, "Fitting the nuisance parameters..."),
        (:info, "→ Reusing previous P(T|W)"),
        (:info, "→ Reusing previous Encoder"),
        (:info, "→ Fitting E[Y|X]"),
        (:info, "Targeting the nuisance parameters..."),
        (:info, "Done.")
    )
    tmle_result, cache = @test_logs log_sequence... tmle!(cache, Ψ, verbosity=1);
    Ψ₀ = -1
    @test covers(tmle_result, Ψ₀)
    #@test closer_than_initial(tmle_result, initial_result, Ψ₀)

    # Change the treatment
    # This will trigger the refit of all η
    Ψ = ATE(
        target=:y₁,
        treatment=(T₂=(case=true, control=false),),
        confounders=[:W₁, :W₂],
    )
    log_sequence = (
        (:info, "Fitting the nuisance parameters..."),
        (:info, "→ Fitting P(T|W)"),
        (:info, "→ Fitting Encoder"),
        (:info, "→ Fitting E[Y|X]"),
        (:info, "Targeting the nuisance parameters..."),
        (:info, "Done.")
    )
    tmle_result, cache = @test_logs log_sequence... tmle!(cache, Ψ, verbosity=1);
    Ψ₀ = 0.5
    @test covers(tmle_result, Ψ₀)
    @test closer_than_initial(tmle_result, Ψ₀)

    # Remove a confounding variable
    # This will trigger the refit of Q and G
    # Since we are not accounting for all confounders 
    # we can't have ground truth coverage on this setting
    Ψ = ATE(
        target=:y₁,
        treatment=(T₂=(case=true, control=false),),
        confounders=[:W₁],
    )
    log_sequence = (
        (:info, "Fitting the nuisance parameters..."),
        (:info, "→ Fitting P(T|W)"),
        (:info, "→ Reusing previous Encoder"),
        (:info, "→ Fitting E[Y|X]"),
        (:info, "Targeting the nuisance parameters..."),
        (:info, "Done.")
    )
    tmle_result, cache = @test_logs log_sequence... tmle!(cache, Ψ, verbosity=1);
    
    # Change the target
    # This will trigger the refit of Q only
    Ψ = ATE(
        target=:y₂,
        treatment=(T₂=(case=true, control=false),),
        confounders=[:W₁],
    )
    log_sequence = (
        (:info, "Fitting the nuisance parameters..."),
        (:info, "→ Reusing previous P(T|W)"),
        (:info, "→ Reusing previous Encoder"),
        (:info, "→ Fitting E[Y|X]"),
        (:info, "Targeting the nuisance parameters..."),
        (:info, "Done.")
    )
    tmle_result, cache = @test_logs log_sequence... tmle!(cache, Ψ, verbosity=1);
    Ψ₀ = 10
    @test covers(tmle_result, Ψ₀)
    @test closer_than_initial(tmle_result, Ψ₀)

end

@testset "Test Warm restart: ATE multiple treatment, $tt" for tt in table_types
    # Note a larger sample size seems necessary here
    dataset = tt(build_dataset(;n=10000))
    # Define the parameter of interest
    Ψ = ATE(
        target=:y₁,
        treatment=(T₁=(case=true, control=false), T₂=(case=true, control=false)),
        confounders=[:W₁, :W₂],
        covariates=[:C₁]
    )
    # Define the nuisance parameters specification
    η_spec = NuisanceSpec(
        LinearRegressor(),
        LogisticClassifier(lambda=0)
    )
    tmle_result, cache = tmle(Ψ, η_spec, dataset; verbosity=0);

    Ψ₀ = -0.5
    @test covers(tmle_result, Ψ₀)
    @test closer_than_initial(tmle_result, Ψ₀)

    # Let's switch case and control for T₂
    Ψ = ATE(
        target=:y₁,
        treatment=(T₁=(case=true, control=false), T₂=(case=false, control=true)),
        confounders=[:W₁, :W₂],
        covariates=[:C₁]
    )
    log_sequence = (
        (:info, "Fitting the nuisance parameters..."),
        (:info, "→ Reusing previous P(T|W)"),
        (:info, "→ Reusing previous Encoder"),
        (:info, "→ Reusing previous E[Y|X]"),
        (:info, "Targeting the nuisance parameters..."),
        (:info, "Done.")
    )
    tmle_result, cache = @test_logs log_sequence... tmle!(cache, Ψ, verbosity=1);

    Ψ₀ = -1.5
    @test covers(tmle_result, Ψ₀)
    @test closer_than_initial(tmle_result, Ψ₀)

end

@testset "Test Warm restart: CM, $tt" for tt in table_types
    dataset = tt(build_dataset(;n=10000))
    Ψ = CM(
        target=:y₁,
        treatment=(T₁=true, T₂=true),
        confounders=[:W₁, :W₂],
        covariates=[:C₁]
    )
    η_spec = NuisanceSpec(
        LinearRegressor(),
        LogisticClassifier(lambda=0)
    )
    tmle_result, cache = tmle(Ψ, η_spec, dataset; verbosity=0);

    Ψ₀ = 3
    @test covers(tmle_result, Ψ₀)
    @test closer_than_initial(tmle_result, Ψ₀)

    # Let's switch case and control for T₂
    Ψ = CM(
        target=:y₁,
        treatment=(T₁=true, T₂=false),
        confounders=[:W₁, :W₂],
        covariates=[:C₁]
    )
    log_sequence = (
        (:info, "Fitting the nuisance parameters..."),
        (:info, "→ Reusing previous P(T|W)"),
        (:info, "→ Reusing previous Encoder"),
        (:info, "→ Reusing previous E[Y|X]"),
        (:info, "Targeting the nuisance parameters..."),
        (:info, "Done.")
    )
    tmle_result, cache = @test_logs log_sequence... tmle!(cache, Ψ, verbosity=1);

    Ψ₀ = 2.5
    @test covers(tmle_result, Ψ₀)
    #@test closer_than_initial(tmle_result, initial_result, Ψ₀)

    # Change the target
    Ψ = CM(
        target=:y₂,
        treatment=(T₁=true, T₂=false),
        confounders=[:W₁, :W₂],
        covariates=[:C₁]
    )
    log_sequence = (
        (:info, "Fitting the nuisance parameters..."),
        (:info, "→ Reusing previous P(T|W)"),
        (:info, "→ Reusing previous Encoder"),
        (:info, "→ Fitting E[Y|X]"),
        (:info, "Targeting the nuisance parameters..."),
        (:info, "Done.")
    )
    tmle_result, cache = @test_logs log_sequence... tmle!(cache, Ψ, verbosity=1);

    Ψ₀ = 1
    @test covers(tmle_result, Ψ₀)
    #@test closer_than_initial(tmle_result, initial_result, Ψ₀)

    # Change the treatment
    Ψ = CM(
        target=:y₂,
        treatment=(T₂=true,),
        confounders=[:W₁, :W₂],
        covariates=[:C₁]
    )
    log_sequence = (
        (:info, "Fitting the nuisance parameters..."),
        (:info, "→ Fitting P(T|W)"),
        (:info, "→ Fitting Encoder"),
        (:info, "→ Fitting E[Y|X]"),
        (:info, "Targeting the nuisance parameters..."),
        (:info, "Done.")
    )
    tmle_result, cache = @test_logs log_sequence... tmle!(cache, Ψ, verbosity=1);

    Ψ₀ = 11
    @test covers(tmle_result, Ψ₀)
    @test closer_than_initial(tmle_result, Ψ₀)
end

@testset "Test Warm restart: pairwise IATE, $tt" for tt in table_types
    dataset = tt(build_dataset(;n=10000))
    Ψ = IATE(
        target=:y₃,
        treatment=(T₁=(case=true, control=false), T₂=(case=true, control=false)),
        confounders=[:W₁, :W₂],
        covariates=[:C₁]
    )
    η_spec = NuisanceSpec(
        LinearRegressor(),
        LogisticClassifier(lambda=0)
    )
    tmle_result, cache = tmle(Ψ, η_spec, dataset; verbosity=0);

    Ψ₀ = -1
    @test covers(tmle_result, Ψ₀)
    @test closer_than_initial(tmle_result, Ψ₀)

    # Remove covariate from fit
    Ψ = IATE(
        target=:y₃,
        treatment=(T₁=(case=true, control=false), T₂=(case=true, control=false)),
        confounders=[:W₁, :W₂],
    )
    log_sequence = (
        (:info, "Fitting the nuisance parameters..."),
        (:info, "→ Reusing previous P(T|W)"),
        (:info, "→ Reusing previous Encoder"),
        (:info, "→ Fitting E[Y|X]"),
        (:info, "Targeting the nuisance parameters..."),
        (:info, "Done.")
    )
    tmle_result, cache = @test_logs log_sequence... tmle!(cache, Ψ, verbosity=1);

    @test covers(tmle_result, Ψ₀)
    @test closer_than_initial(tmle_result, Ψ₀)

    # Changing the treatments values
    Ψ = IATE(
        target=:y₃,
        treatment=(T₁=(case=false, control=true), T₂=(case=true, control=false)),
        confounders=[:W₁, :W₂],
    )
    log_sequence = (
        (:info, "Fitting the nuisance parameters..."),
        (:info, "→ Reusing previous P(T|W)"),
        (:info, "→ Reusing previous Encoder"),
        (:info, "→ Reusing previous E[Y|X]"),
        (:info, "Targeting the nuisance parameters..."),
        (:info, "Done.")
    )
    tmle_result, cache = @test_logs log_sequence... tmle!(cache, Ψ, verbosity=1);

    Ψ₀ = - Ψ₀
    @test covers(tmle_result, Ψ₀)
    @test closer_than_initial(tmle_result, Ψ₀)

end

@testset "Test Warm restart: Both Ψ and η changed" begin
    dataset = build_dataset(;n=10000)
    # Define the parameter of interest
    Ψ = ATE(
        target=:y₁,
        treatment=(T₁=(case=true, control=false), T₂=(case=true, control=false)),
        confounders=[:W₁, :W₂],
        covariates=[:C₁]
    )
    # Define the nuisance parameters specification
    η_spec = NuisanceSpec(
        LinearRegressor(),
        LogisticClassifier(lambda=0)
    )
    tmle_result, cache = tmle(Ψ, η_spec, dataset; verbosity=0);
    Ψ₀ = -0.5
    @test covers(tmle_result, Ψ₀)

    Ψnew = ATE(
        target=:y₁,
        treatment=(T₁=(case=false, control=true), T₂=(case=false, control=true)),
        confounders=[:W₁, :W₂],
        covariates=[:C₁]
    )
    # Define the nuisance parameters specification
    η_spec_new = NuisanceSpec(
        LinearRegressor(),
        LogisticClassifier(lambda=0.1)
    )
    log_sequence = (
        (:info, "Fitting the nuisance parameters..."),
        (:info, "→ Reusing previous P(T|W)"),
        (:info, "→ Reusing previous Encoder"),
        (:info, "→ Reusing previous E[Y|X]"),
        (:info, "Targeting the nuisance parameters..."),
        (:info, "Done.")
    )
    tmle_result, cache = @test_logs log_sequence... tmle!(cache, Ψnew, η_spec; verbosity=1);
    Ψ₀ = 0.5
    @test covers(tmle_result, Ψ₀)
end

end

true