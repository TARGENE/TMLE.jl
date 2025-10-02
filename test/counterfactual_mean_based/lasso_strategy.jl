using Test
using TMLE
using DataFrames
using CategoricalArrays
using Random

function create_test_data(n=100, p=5)
    Random.seed!(123)
    W_df = DataFrame(Dict(Symbol("W$i") => randn(n) for i in 1:p))
    A_vals = rand([0, 1], n)
    A = categorical(A_vals; levels=[0, 1])
    Y = 2.0 * A_vals + sum(Matrix(W_df[:, 1:3]), dims=2)[:, 1] + randn(n) * 0.3
    return hcat(W_df, DataFrame(A=A, Y=Y))
end

@testset "LASSO Collaborative TMLE" begin
    
    @testset "Basic construction and defaults" begin
        strategy = LassoCTMLE(confounders=[:W1, :W2, :W3])
        @test strategy.confounders == [:W1, :W2, :W3]
        @test strategy.patience == 5
        @test length(strategy.lambda_path) == 0 
        @test strategy.alpha == 1.0 
        @test strategy.current_iteration == 0
    end

    @testset "LASSO CTMLE with automatic CV lambda selection" begin
        dataset = create_test_data(150, 8)
        confounders = [Symbol("W$i") for i in 1:8]
        
        estimand = ATE(
            outcome = :Y,
            treatment_values = (A = (case = 1, control = 0),),
            treatment_confounders = (A = confounders,)
        )
        
        # Test LASSO CTMLE with default settings (automatic CV lambda)
        lasso_strategy = LassoCTMLE(confounders = confounders)
        lasso_estimator = Tmle(collaborative_strategy = lasso_strategy)
        lasso_result, _ = lasso_estimator(estimand, dataset; verbosity = 0)
        
        @test !isnan(estimate(lasso_result))
        
        # Compare with standard TMLE to ensure regularization works
        standard_estimator = Tmle()
        standard_result, _ = standard_estimator(estimand, dataset; verbosity = 0)
        
        @test !isnan(estimate(standard_result))
        @test estimate(lasso_result) != estimate(standard_result) 
    end
    
end
