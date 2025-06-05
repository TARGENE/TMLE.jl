using Test
using TMLE

TEST_DIR = joinpath(pkgdir(TMLE), "test")

@time begin
    # Test general functionality
    @test include(joinpath(TEST_DIR, "utils.jl"))
    @test include(joinpath(TEST_DIR, "scm.jl"))
    @test include(joinpath(TEST_DIR, "adjustment.jl"))
    @test include(joinpath(TEST_DIR, "estimands.jl"))
    @test include(joinpath(TEST_DIR, "estimators_and_estimates.jl"))
    @test include(joinpath(TEST_DIR, "missing_management.jl"))
    @test include(joinpath(TEST_DIR, "composition.jl"))
    @test include(joinpath(TEST_DIR, "resampling.jl"))
    
    # Test Counterfactual Mean Based Estimation
    @test include(joinpath(TEST_DIR, "counterfactual_mean_based/estimands.jl"))
    @test include(joinpath(TEST_DIR, "counterfactual_mean_based/clever_covariate.jl"))
    @test include(joinpath(TEST_DIR, "counterfactual_mean_based/gradient.jl"))
    @test include(joinpath(TEST_DIR, "counterfactual_mean_based/fluctuation.jl"))
    @test include(joinpath(TEST_DIR, "counterfactual_mean_based/estimators_and_estimates.jl"))
    @test include(joinpath(TEST_DIR, "counterfactual_mean_based/non_regression_test.jl"))
    @test include(joinpath(TEST_DIR, "counterfactual_mean_based/double_robustness_ate.jl"))
    @test include(joinpath(TEST_DIR, "counterfactual_mean_based/double_robustness_aie.jl"))
    @test include(joinpath(TEST_DIR, "counterfactual_mean_based/3points_interactions.jl"))
    @test include(joinpath(TEST_DIR, "counterfactual_mean_based/collaborative_template.jl"))
    @test include(joinpath(TEST_DIR, "counterfactual_mean_based/covariate_based_strategies.jl"))
    
    # Test Extensions
    if VERSION >= v"1.9"
        @test include(joinpath(TEST_DIR, "configuration.jl"))
        @test include(joinpath(TEST_DIR, "causaltables_interface.jl"))
    end

    # Test Experimental
    @test include(joinpath(TEST_DIR, "estimand_ordering.jl"))
end