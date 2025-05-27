module TestInteractionATE

using Test
using TMLE
using Tables
using MLJModels
using MLJLinearModels
using MLJXGBoostInterface

TEST_DIR = joinpath(pkgdir(TMLE), "test")

include(joinpath(TEST_DIR, "helper_fns.jl"))
include(joinpath(TEST_DIR, "counterfactual_mean_based", "interactions_simulations.jl"))

cont_interacter = InteractionTransformer(order=2) |> LinearRegressor
cat_interacter = InteractionTransformer(order=2) |> LogisticClassifier(lambda=1.)


@testset "Test Double Robustness AIE on binary_outcome_binary_treatment_pb" begin
    dataset, Ψ₀ = binary_outcome_binary_treatment_pb(n=10_000)
    Ψ = AIE(
        outcome=:Y,
        treatment_values = (
            T₁=(case=true, control=false), 
            T₂=(case=true, control=false)
        ),
        treatment_confounders = (
            T₁=[:W₁, :W₂, :W₃],
            T₂=[:W₁, :W₂, :W₃],
        )
    )
    # When Q is misspecified but G is well specified
    # Note that LogisticClassifiers are not enough to recapitulate the generative multinomial here
    models = Dict(
        :Y  => with_encoder(ConstantClassifier()),
        :T₁ => with_encoder(XGBoostClassifier(;nthread=1)),
        :T₂ => with_encoder(XGBoostClassifier(;nthread=1)),
    )
    dr_estimators = double_robust_estimators(models, resampling=StratifiedCV())
    results, cache = test_coverage_and_get_results(dr_estimators, Ψ, Ψ₀, dataset; verbosity=0)
    test_mean_inf_curve_almost_zero(results.tmle; atol=1e-9)
    test_mean_inf_curve_almost_zero(results.ose; atol=1e-9)
    # The initial estimate is far away
    naive = Plugin(models[:Y])
    naive_result, cache = naive(Ψ, dataset; cache=cache, verbosity=0)
    @test naive_result == 0

    # When Q is well specified  but G is misspecified
    models = Dict(
        :Y  => with_encoder(LogisticClassifier(lambda=0)),
        :T₁ => with_encoder(ConstantClassifier()),
        :T₂ => with_encoder(ConstantClassifier()),
    )
    dr_estimators = double_robust_estimators(models, resampling=StratifiedCV())
    results, cache = test_coverage_and_get_results(dr_estimators, Ψ, Ψ₀, dataset; verbosity=0)
    test_mean_inf_curve_almost_zero(results.tmle; atol=1e-9)
    test_mean_inf_curve_almost_zero(results.ose; atol=1e-9)
    # The initial estimate is far away
    naive = Plugin(models[:Y])
    naive_result, cache = naive(Ψ, dataset; cache=cache, verbosity=0)
    @test naive_result ≈ -0.0 atol=1e-1
end

@testset "Test Double Robustness AIE on continuous_outcome_binary_treatment_pb" begin
    dataset, Ψ₀ = continuous_outcome_binary_treatment_pb(n=10_000)
    Ψ = AIE(
        outcome = :Y,
        treatment_values = (
            T₁=(case=true, control=false), 
            T₂=(case=true, control=false)
        ),
        treatment_confounders = (
            T₁=[:W₁, :W₂, :W₃],
            T₂=[:W₁, :W₂, :W₃],
        )
    )
    # When Q is misspecified but G is well specified
    # Note that LogisticClassifiers are not enough to recapitulate the generative multinomial here
    models = Dict(
        :Y  => with_encoder(MLJModels.DeterministicConstantRegressor()),
        :T₁ => with_encoder(XGBoostClassifier(;nthread=1)),
        :T₂ => with_encoder(XGBoostClassifier(;nthread=1)),
    )

    dr_estimators = double_robust_estimators(models)
    results, cache = test_coverage_and_get_results(dr_estimators, Ψ, Ψ₀, dataset; verbosity=0)
    test_mean_inf_curve_almost_zero(results.tmle; atol=1e-10)
    test_mean_inf_curve_almost_zero(results.ose; atol=1e-10)
    # The initial estimate is far away
    naive = Plugin(models[:Y])
    naive_result, cache = naive(Ψ, dataset; cache=cache, verbosity=0)
    @test naive_result == 0

    # When Q is well specified  but G is misspecified
    models = Dict(
        :Y  => with_encoder(cont_interacter),
        :T₁ => with_encoder(ConstantClassifier()),
        :T₂ => with_encoder(ConstantClassifier()),
    )
    dr_estimators = double_robust_estimators(models)
    results, cache = test_coverage_and_get_results(dr_estimators, Ψ, Ψ₀, dataset; verbosity=0)
    test_mean_inf_curve_almost_zero(results.tmle; atol=1e-10)
end


@testset "Test Double Robustness AIE on binary_outcome_categorical_treatment_pb" begin
    dataset, Ψ₀ = binary_outcome_categorical_treatment_pb(n=30_000)
    Ψ = AIE(
        outcome=:Y,
        treatment_values= (
            T₁=(case="CC", control="CG"), 
            T₂=(case="AT", control="AA")
        ),
        treatment_confounders = (
            T₁=[:W₁, :W₂, :W₃],
            T₂=[:W₁, :W₂, :W₃],
        )
    )
    # When Q is misspecified but G is well specified
    # Note that LogisticClassifiers are not enough to recapitulate the generative multinomial here
    models = Dict(
        :Y  => with_encoder(ConstantClassifier()),
        :T₁ => with_encoder(XGBoostClassifier(;nthread=1)),
        :T₂ => with_encoder(XGBoostClassifier(;nthread=1))
    )
    dr_estimators = double_robust_estimators(models, resampling=StratifiedCV())
    results, cache = test_coverage_and_get_results(dr_estimators, Ψ, Ψ₀, dataset; verbosity=0)
    test_mean_inf_curve_almost_zero(results.tmle; atol=1e-5)
    test_mean_inf_curve_almost_zero(results.ose; atol=1e-10)

    # The initial estimate is far away
    naive = Plugin(models[:Y])
    naive_result, cache = naive(Ψ, dataset; cache=cache, verbosity=0)
    @test naive_result == 0

    # When Q is well specified but G is misspecified
    models = Dict(
        :Y  => with_encoder(cat_interacter),
        :T₁ => with_encoder(ConstantClassifier()),
        :T₂ => with_encoder(ConstantClassifier()),
    )
    dr_estimators = double_robust_estimators(models, resampling=StratifiedCV())
    results, cache = test_coverage_and_get_results(dr_estimators, Ψ, Ψ₀, dataset; verbosity=0)
    test_mean_inf_curve_almost_zero(results.tmle; atol=1e-5)
    test_mean_inf_curve_almost_zero(results.ose; atol=1e-10)

    # The initial estimate is far away
    naive = Plugin(models[:Y])
    naive_result, cache = naive(Ψ, dataset; cache=cache, verbosity=0)
    @test naive_result ≈ -0.02 atol=1e-2
end


end;


true