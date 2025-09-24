using Test
using TMLE

const TEST_DIR = joinpath(pkgdir(TMLE), "test")
@testset "TMLE.jl" begin
        # Test general functionality
        include("utils.jl")
        include("scm.jl")
        include("adjustment.jl")
        include("estimands.jl")
        include("estimators_and_estimates.jl")
        include("missing_management.jl")
        include("composition.jl")
        include("resampling.jl")

        # Test Counterfactual Mean Based Estimation
        include("counterfactual_mean_based/estimands.jl")
        include("counterfactual_mean_based/clever_covariate.jl")
        include("counterfactual_mean_based/gradient.jl")
        include("counterfactual_mean_based/fluctuation.jl")
        include("counterfactual_mean_based/estimators_and_estimates.jl")
        include("counterfactual_mean_based/non_regression_test.jl")
        include("counterfactual_mean_based/double_robustness_ate.jl")
        include("counterfactual_mean_based/double_robustness_aie.jl")
        include("counterfactual_mean_based/3points_interactions.jl")
        include("counterfactual_mean_based/collaborative_template.jl")
        include("counterfactual_mean_based/covariate_based_strategies.jl")
        include("counterfactual_mean_based/lasso_strategy.jl")

        # Test Extensions
        if VERSION >= v"1.9"
            include("configuration.jl")
            include("causaltables_interface.jl")
        end

        # Test Experimental
        include("estimand_ordering.jl")
end
