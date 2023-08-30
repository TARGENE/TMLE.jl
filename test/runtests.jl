using Test

@time begin
    @test include("distribution_factors.jl")
    @test include("scm.jl")
    @test include("utils.jl")
    @test include("missing_management.jl")
    @test include("composition.jl")
    @test include("treatment_transformer.jl")
    @test include("counterfactual_mean_based/non_regression_test.jl")
    @test include("counterfactual_mean_based/offset_and_covariate.jl")
    @test include("counterfactual_mean_based/gradient.jl")
    @test include("counterfactual_mean_based/fluctuation.jl")
    @test include("counterfactual_mean_based/double_robustness_ate.jl")
    @test include("counterfactual_mean_based/double_robustness_iate.jl")
    @test include("counterfactual_mean_based/3points_interactions.jl")
    @test include("counterfactual_mean_based/estimators.jl")
    @test include("counterfactual_mean_based/estimands.jl")
end