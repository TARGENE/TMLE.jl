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
using LogExpFunctions
using DataFrames

TESTDIR = joinpath(pkgdir(TMLE), "test")
include(joinpath(TESTDIR, "helper_fns.jl"))
include(joinpath(TESTDIR, "counterfactual_mean_based", "ate_simulations.jl"))


@testset "Test Double Robustness ATE on continuous_outcome_categorical_treatment_pb" begin
    dataset, Ψ₀ = continuous_outcome_categorical_treatment_pb(;n=10_000, control="TT", case="AA")
    Ψ = ATE(
        outcome   = :Y,
        treatment_values = (T=(case="AA", control="TT"),),
        treatment_confounders = (T = [:W₁, :W₂, :W₃],)
    )
    
    # When Q is misspecified but G is well specified
    models = Dict(
        :Y => with_encoder(MLJModels.DeterministicConstantRegressor()),
        :T => with_encoder(LogisticClassifier(lambda=0))
    )
    dr_estimators = double_robust_estimators(models)
    results, cache = test_coverage_and_get_results(dr_estimators, Ψ, Ψ₀, dataset; verbosity=0)
    test_mean_inf_curve_almost_zero(results.tmle; atol=1e-10)
    test_mean_inf_curve_almost_zero(results.ose; atol=1e-10)
    # Test emptyIC function
    @test emptyIC(results.tmle).IC == []
    pval = pvalue(OneSampleZTest(results.tmle))
    @test emptyIC(results.tmle, pval_threshold=0.9pval).IC == []
    @test emptyIC(results.tmle, pval_threshold=1.1pval) === results.tmle
    # The initial estimate is far away
    naive = Plugin(models[:Y])
    naive_result, cache = naive(Ψ, dataset; cache=cache, verbosity=0)
    @test naive_result == 0
    
    # When Q is well specified but G is misspecified
    models = Dict(
        :Y => with_encoder(LinearRegressor()),
        :T => with_encoder(ConstantClassifier())
    )
    dr_estimators = double_robust_estimators(models)
    results, cache = test_coverage_and_get_results(dr_estimators, Ψ, Ψ₀, dataset; verbosity=0)
    test_mean_inf_curve_almost_zero(results.tmle; atol=1e-10)
end

@testset "Test Double Robustness ATE on binary_outcome_binary_treatment_pb" begin
    dataset, Ψ₀ = binary_outcome_binary_treatment_pb(;n=10_000)
    Ψ = ATE(
        outcome = :Y,
        treatment_values = (T=(case=true, control=false),),
        treatment_confounders = (T=[:W],)
    )
    # When Q is misspecified but G is well specified
    models = Dict(
        :Y => with_encoder(ConstantClassifier()),
        :T => with_encoder(LogisticClassifier(lambda=0))
    )
    dr_estimators = double_robust_estimators(models, resampling=StratifiedCV())
    results, cache = test_coverage_and_get_results(dr_estimators, Ψ, Ψ₀, dataset; verbosity=0)
    test_mean_inf_curve_almost_zero(results.tmle; atol=1e-6)
    test_mean_inf_curve_almost_zero(results.ose; atol=1e-6)
    # The initial estimate is far away
    naive = Plugin(models[:Y])
    naive_result, cache = naive(Ψ, dataset; cache=cache, verbosity=0) 
    @test naive_result == 0
    # When Q is well specified but G is misspecified
    models = Dict(
        :Y => with_encoder(LogisticClassifier(lambda=0)),
        :T => with_encoder(ConstantClassifier())
    )
    dr_estimators = double_robust_estimators(models, resampling=StratifiedCV())
    results, cache = test_coverage_and_get_results(dr_estimators, Ψ, Ψ₀, dataset; verbosity=0)
    test_mean_inf_curve_almost_zero(results.tmle; atol=1e-6)
end


@testset "Test Double Robustness ATE on continuous_outcome_binary_treatment_pb" begin
    dataset, Ψ₀ = continuous_outcome_binary_treatment_pb(n=50_000)
    Ψ = ATE(
        outcome = :Y,
        treatment_values = (T=(case=true, control=false),),
        treatment_confounders = (T=[:W₁, :W₂, :W₃],)
    )
    # When Q is misspecified but G is well specified
    models = Dict(
        :Y => with_encoder(MLJModels.DeterministicConstantRegressor()),
        :T => with_encoder(LogisticClassifier(lambda=0))
    )
    dr_estimators = double_robust_estimators(models)
    results, cache = test_coverage_and_get_results(dr_estimators, Ψ, Ψ₀, dataset; verbosity=0)

    test_mean_inf_curve_almost_zero(results.tmle; atol=1e-10)
    test_mean_inf_curve_almost_zero(results.ose; atol=1e-10)
    # The initial estimate is far away
    naive = Plugin(models[:Y])
    naive_result, cache = naive(Ψ, dataset; cache=cache, verbosity=0)
    @test naive_result == 0

    # When Q is well specified but G is misspecified
    models = Dict(
        :Y => with_encoder(LinearRegressor()),
        :T => with_encoder(ConstantClassifier())
    )
    dr_estimators = double_robust_estimators(models)
    results, cache = test_coverage_and_get_results(dr_estimators, Ψ, Ψ₀, dataset; verbosity=0)
    test_mean_inf_curve_almost_zero(results.tmle; atol=1e-10)
end

@testset "Test Double Robustness ATE with two treatment variables" begin
    dataset, (ATE₁₁₋₀₁, ATE₁₁₋₀₀) = dataset_2_treatments_pb(;rng = StableRNG(123), n=50_000)
    # Test first ATE, only T₁ treatment varies 
    Ψ = ATE(
        outcome = :Y,
        treatment_values = (
            T₁=(case=1., control=0.), 
            T₂=(case=1., control=1.)
        ),
        treatment_confounders = (
            T₁ = [:W₁, :W₂],
            T₂ = [:W₁, :W₂],
        )
    )
    # When Q is misspecified but G is well specified
    models = Dict(
        :Y  => with_encoder(MLJModels.DeterministicConstantRegressor()),
        :T₁ => with_encoder(LogisticClassifier(lambda=0)),
        :T₂ => with_encoder(LogisticClassifier(lambda=0))
    )
    dr_estimators = double_robust_estimators(models)
    results, cache = test_coverage_and_get_results(dr_estimators, Ψ, ATE₁₁₋₀₁, dataset; verbosity=0)
    test_mean_inf_curve_almost_zero(results.tmle; atol=1e-10)
    test_mean_inf_curve_almost_zero(results.ose; atol=1e-10)
    # When Q is well specified but G is misspecified
    models = Dict(
        :Y  => with_encoder(LinearRegressor()),
        :T₁ => with_encoder(ConstantClassifier()),
        :T₂ => with_encoder(ConstantClassifier())
    )
    dr_estimators = double_robust_estimators(models)
    results, cache = test_coverage_and_get_results(dr_estimators, Ψ, ATE₁₁₋₀₁, dataset; verbosity=0)
    test_mean_inf_curve_almost_zero(results.tmle; atol=1e-10)

    # Test second ATE, two treatment varies 
    Ψ = ATE(
        outcome = :Y,
        treatment_values = (
            T₁=(case=1., control=0.), 
            T₂=(case=1., control=0.)
        ),
        treatment_confounders = (
            T₁ = [:W₁, :W₂],
            T₂ = [:W₁, :W₂],
        )
    )
    dr_estimators = double_robust_estimators(models)
    results, cache = test_coverage_and_get_results(dr_estimators, Ψ, ATE₁₁₋₀₀, dataset; verbosity=0)
    test_mean_inf_curve_almost_zero(results.tmle; atol=1e-10)
    test_mean_inf_curve_almost_zero(results.ose; atol=1e-10)

    # When Q is well specified but G is misspecified
    models = Dict(
        :Y  => with_encoder(MLJModels.DeterministicConstantRegressor()),
        :T₁ => with_encoder(LogisticClassifier(lambda=0)),
        :T₂ => with_encoder(LogisticClassifier(lambda=0)),
    )
    dr_estimators = double_robust_estimators(models)
    results, cache = test_coverage_and_get_results(dr_estimators, Ψ, ATE₁₁₋₀₀, dataset; verbosity=0)
    test_mean_inf_curve_almost_zero(results.tmle; atol=1e-10)
end

end;

true