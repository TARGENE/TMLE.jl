module TestEstimation

using Test
using Tables
using StableRNGs
using TMLE
using DataFrames
using Distributions
using MLJLinearModels
using CategoricalArrays
using LogExpFunctions

include(joinpath(dirname(@__DIR__), "helper_fns.jl"))

"""
Results derived by hand for this dataset:

- ATE(y₁, T₁, W₁, W₂, C₁) = 1 - 4E[C₁] = 1 - 2 = -1
- ATE(y₁, T₂, W₁, W₂, C₁) = E[W₂] = 0.5
- ATE(y₁, (T₁, T₂), W₁, W₂, C₁) = E[-4C₁ + 1 + W₂] = -0.5
- ATE(y₂, T₂, W₁, W₂, C₁) = 10
"""
function build_dataset_and_scm(;n=100)
    rng = StableRNG(123)
    # Confounders
    W₁ = rand(rng, Uniform(), n)
    W₂ = rand(rng, Uniform(), n)
    # Covariates
    C₁ = rand(rng, n)
    # Treatment | Confounders
    T₁ = rand(rng, Uniform(), n) .< logistic.(0.5sin.(W₁) .- 1.5W₂)
    T₂ = rand(rng, Uniform(), n) .< logistic.(-3W₁ - 1.5W₂)
    # outcome | Confounders, Covariates, Treatments
    Y₁ = 1 .+ 2W₁ .+ 3W₂ .- 4C₁.*T₁ .+ T₁ + T₂.*W₂ .+ rand(rng, Normal(0, 0.1), n)
    Y₂ = 1 .+ 10T₂ .+ W₂ .+ rand(rng, Normal(0, 0.1), n)
    Y₃ = 1 .+ 2W₁ .+ 3W₂ .- 4C₁.*T₁ .- 2T₂.*T₁.*W₂ .+ rand(rng, Normal(0, 0.1), n)
    dataset = (
        T₁ = categorical(T₁),
        T₂ = categorical(T₂),
        W₁ = W₁, 
        W₂ = W₂,
        C₁ = C₁,
        Y₁ = Y₁,
        Y₂ = Y₂,
        Y₃ = Y₃
        )
    scm = SCM(
        SE(:T₁, [:W₁, :W₂], LogisticClassifier(lambda=0)),
        SE(:T₂, [:W₁, :W₂], LogisticClassifier(lambda=0)),
        SE(:Y₁, [:T₁, :T₂, :W₁, :W₂, :C₁], TreatmentTransformer() |> LinearRegressor()),
        SE(:Y₂, [:T₂, :W₂], TreatmentTransformer() |> LinearRegressor()),
        SE(:Y₃, [:T₁, :T₂, :W₁, :W₂, :C₁], TreatmentTransformer() |> LinearRegressor()),
    )
    return dataset, scm
end


table_types = (Tables.columntable, DataFrame)

@testset "Test Warm restart: ATE single treatment, $tt" for tt in table_types
    dataset, scm = build_dataset_and_scm(;n=50_000)
    dataset = tt(dataset)
    # Define the estimand of interest
    Ψ = ATE(
        scm,
        outcome=:Y₁,
        treatment=(T₁=(case=true, control=false),),
    )
    # Run TMLE
    log_sequence = (
        (:info, "Fitting the required equations..."),
        (:info, "Fitting Structural Equation corresponding to variable T₁."),
        (:info, "Fitting Structural Equation corresponding to variable Y₁."),
        (:info, "Performing TMLE..."),
        (:info, "Done.")
    )
    tmle_result, fluctuation_mach = @test_logs log_sequence... tmle!(Ψ, dataset; verbosity=1);
    # The TMLE covers the ground truth but the initial estimate does not
    Ψ₀ = -1
    test_coverage(tmle_result, Ψ₀)
    test_fluct_decreases_risk(Ψ, fluctuation_mach)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-10)

    # Update the treatment specification
    # Nuisance estimands should not fitted again
    Ψ = ATE(
        scm,
        outcome=:Y₁,
        treatment=(T₁=(case=false, control=true),),
    )
    log_sequence = (
        (:info, "Fitting the required equations..."),
        (:info, "Performing TMLE..."),
        (:info, "Done.")
    )
    tmle_result, fluctuation_mach = @test_logs log_sequence... tmle!(Ψ, dataset, verbosity=1);
    Ψ₀ = 1
    test_coverage(tmle_result, Ψ₀)
    test_fluct_decreases_risk(Ψ, fluctuation_mach)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-10)

    # New marginal treatment estimation: T₂
    # This will trigger fit of the corresponding equations
    Ψ = ATE(
        scm,
        outcome=:Y₁,
        treatment=(T₂=(case=true, control=false),),
    )
    log_sequence = (
        (:info, "Fitting the required equations..."),
        (:info, "Fitting Structural Equation corresponding to variable T₂."),
        (:info, "Fitting Structural Equation corresponding to variable Y₁."),
        (:info, "Performing TMLE..."),
        (:info, "Done.")
    )
    tmle_result, fluctuation_mach = @test_logs log_sequence... tmle!(Ψ, dataset, verbosity=1);
    Ψ₀ = 0.5
    test_coverage(tmle_result, Ψ₀)
    test_fluct_decreases_risk(Ψ, fluctuation_mach)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-10)

    # Change the outcome: Y₂
    # This will trigger the fit for the corresponding equation
    Ψ = ATE(
        scm,
        outcome=:Y₂,
        treatment=(T₂=(case=true, control=false),),
    )
    log_sequence = (
        (:info, "Fitting the required equations..."),
        (:info, "Fitting Structural Equation corresponding to variable Y₂."),
        (:info, "Performing TMLE..."),
        (:info, "Done.")
    )
    tmle_result, fluctuation_mach = @test_logs log_sequence... tmle!(Ψ, dataset, verbosity=1);
    Ψ₀ = 10
    test_coverage(tmle_result, Ψ₀)
    test_fluct_decreases_risk(Ψ, fluctuation_mach)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-10)

    # New estimand with 2 treatment variables
    Ψ = ATE(
        scm,
        outcome=:Y₁,
        treatment=(T₁=(case=true, control=false), T₂=(case=true, control=false)),
    )
    log_sequence = (
        (:info, "Fitting the required equations..."),
        (:info, "Fitting Structural Equation corresponding to variable Y₁."),
        (:info, "Performing TMLE..."),
        (:info, "Done.")
    )
    tmle_result, fluctuation_mach = @test_logs log_sequence... tmle!(Ψ, dataset, verbosity=1);
    Ψ₀ = -0.5
    test_coverage(tmle_result, Ψ₀)
    test_fluct_decreases_risk(Ψ, fluctuation_mach)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-10)

    # Switching the T₂ setting
    Ψ = ATE(
        scm,
        outcome=:Y₁,
        treatment=(T₁=(case=true, control=false), T₂=(case=false, control=true)),
    )
    log_sequence = (
        (:info, "Fitting the required equations..."),
        (:info, "Performing TMLE..."),
        (:info, "Done.")
    )
    tmle_result, fluctuation_mach = @test_logs log_sequence... tmle!(Ψ, dataset, verbosity=1);
    Ψ₀ = -1.5
    test_coverage(tmle_result, Ψ₀)
    test_fluct_decreases_risk(Ψ, fluctuation_mach)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-10)
end

@testset "Test Warm restart: CM, $tt" for tt in table_types
    dataset, scm = build_dataset_and_scm(;n=50_000)
    dataset = tt(dataset)
    
    Ψ = CM(
        scm,
        outcome=:Y₁,
        treatment=(T₁=true, T₂=true),
    )
    log_sequence = (
        (:info, "Fitting the required equations..."),
        (:info, "Fitting Structural Equation corresponding to variable T₁."),
        (:info, "Fitting Structural Equation corresponding to variable T₂."),
        (:info, "Fitting Structural Equation corresponding to variable Y₁."),
        (:info, "Performing TMLE..."),
        (:info, "Done.")
    )
    tmle_result, fluctuation_mach = @test_logs log_sequence... tmle!(Ψ, dataset, verbosity=1);
    Ψ₀ = 3
    test_coverage(tmle_result, Ψ₀)
    test_fluct_decreases_risk(Ψ, fluctuation_mach)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-10)

    # Let's switch case and control for T₂
    Ψ = CM(
        scm,
        outcome=:Y₁,
        treatment=(T₁=true, T₂=false),
    )
    log_sequence = (
        (:info, "Fitting the required equations..."),
        (:info, "Performing TMLE..."),
        (:info, "Done.")
    )
    tmle_result, fluctuation_mach = @test_logs log_sequence... tmle!(Ψ, dataset, verbosity=1);
    Ψ₀ = 2.5
    test_coverage(tmle_result, Ψ₀)
    test_fluct_decreases_risk(Ψ, fluctuation_mach)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-10)

    # Change the outcome
    Ψ = CM(
        scm,
        outcome=:Y₂,
        treatment=(T₂=false, ),
    )
    log_sequence = (
        (:info, "Fitting the required equations..."),
        (:info, "Fitting Structural Equation corresponding to variable Y₂."),
        (:info, "Performing TMLE..."),
        (:info, "Done.")
    )
    tmle_result, fluctuation_mach = @test_logs log_sequence... tmle!(Ψ, dataset, verbosity=1);
    Ψ₀ = 1.5
    test_coverage(tmle_result, Ψ₀)
    test_fluct_decreases_risk(Ψ, fluctuation_mach)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-10)

    # Change the treatment model
    scm.T₂.model = LogisticClassifier(lambda=0.01)
    log_sequence = (
        (:info, "Fitting the required equations..."),
        (:info, "Fitting Structural Equation corresponding to variable T₂."),
        (:info, "Performing TMLE..."),
        (:info, "Done.")
    )
    tmle_result, fluctuation_mach = @test_logs log_sequence... tmle!(Ψ, dataset, verbosity=1);
    test_coverage(tmle_result, Ψ₀)
    test_fluct_decreases_risk(Ψ, fluctuation_mach)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-10)
end

@testset "Test Warm restart: pairwise IATE, $tt" for tt in table_types
    dataset, scm = build_dataset_and_scm(;n=50_000)
    dataset = tt(dataset)
    Ψ = IATE(
        scm,
        outcome=:Y₃,
        treatment=(T₁=(case=true, control=false), T₂=(case=true, control=false)),
    )
    log_sequence = (
        (:info, "Fitting the required equations..."),
        (:info, "Fitting Structural Equation corresponding to variable T₁."),
        (:info, "Fitting Structural Equation corresponding to variable T₂."),
        (:info, "Fitting Structural Equation corresponding to variable Y₃."),
        (:info, "Performing TMLE..."),
        (:info, "Done.")
    )
    tmle_result, fluctuation_mach = @test_logs log_sequence... tmle!(Ψ, dataset, verbosity=1);
    Ψ₀ = -1
    test_coverage(tmle_result, Ψ₀)
    test_fluct_decreases_risk(Ψ, fluctuation_mach)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-10)

    # Changing some equations
    scm.Y₃.model = TreatmentTransformer() |> RidgeRegressor(lambda=1e-5)
    scm.T₂.model = LogisticClassifier(lambda=1e-5)

    log_sequence = (
        (:info, "Fitting the required equations..."),
        (:info, "Fitting Structural Equation corresponding to variable T₂."),
        (:info, "Fitting Structural Equation corresponding to variable Y₃."),
        (:info, "Performing TMLE..."),
        (:info, "Done.")
    )
    tmle_result, fluctuation_mach = @test_logs log_sequence... tmle!(Ψ, dataset, verbosity=1);
    test_coverage(tmle_result, Ψ₀)
    test_fluct_decreases_risk(Ψ, fluctuation_mach)
    test_mean_inf_curve_almost_zero(tmle_result; atol=1e-10)
end

end

true