#!/usr/bin/env julia
using Pkg
Pkg.activate(".")

using Test
using TMLE
using DataFrames
using CategoricalArrays
using Random

function create_simple_test_data(n=100)
    Random.seed!(123)
    W1 = randn(n)
    A_vals = rand([0, 1], n)
    A = categorical(A_vals; levels=[0, 1])
    Y = 2.0 * A_vals + 0.5 * W1 + randn(n) * 0.3
    return DataFrame(W1=W1, A=A, Y=Y)
end

@testset "LASSO Collaborative TMLE" begin
    
    @testset "Basic LassoCTMLE construction" begin
        # Test with CV lambda generation
        strategy = LassoCTMLE(confounders=[:W1])
        @test strategy.confounders == [:W1]
        @test strategy.patience == 5
        @test length(strategy.lambda_path) == 0
        @test strategy.cv_folds == 5
        @test strategy.alpha == 1.0
        @test strategy.current_iteration == 0
        @test strategy.explored_lambdas == Set{Float64}()
        @test strategy.best_lambda === nothing
        @test strategy.best_cv_loss == Inf
        
        # Test with manual lambda specification
        manual_strategy = LassoCTMLE(
            confounders=[:W1], 
            lambda_path=[0.1, 0.01, 0.001]
        )
        @test length(manual_strategy.lambda_path) == 3
        @test manual_strategy.lambda_path == [0.1, 0.01, 0.001]
    end

    @testset "Collaborative interface methods" begin
        confounders = [:W1]
        dataset = create_simple_test_data(50)
        
        Ψ = ATE(
            outcome = :Y,
            treatment_values = (A = (case = 1, control = 0),),
            treatment_confounders = (A = confounders,)
        )
        
        # Test CV lambda generation
        strategy = LassoCTMLE(confounders = confounders, patience = 3)
        
        TMLE.initialise!(strategy, Ψ)
        @test !TMLE.exhausted(strategy)
        TMLE.finalise!(strategy)
    end

    @testset "LASSO CTMLE end-to-end" begin
        confounders = [:W1]
        dataset = create_simple_test_data(100)
        
        Ψ = ATE(
            outcome = :Y,
            treatment_values = (A = (case = 1, control = 0),),
            treatment_confounders = (A = confounders,)
        )
        
        strategy = LassoCTMLE(
            confounders = confounders,
            patience = 2,
            lambda_path = [0.1, 0.01, 0.001]
        )
        
        lasso_estimator = Tmle(collaborative_strategy = strategy)
        result, _ = lasso_estimator(Ψ, dataset; verbosity = 0)
        @test !isnan(estimate(result))
    end

    @testset "Compare with Standard TMLE" begin
        confounders = [:W1]
        dataset = create_simple_test_data(100)
        
        Ψ = ATE(
            outcome = :Y,
            treatment_values = (A = (case = 1, control = 0),),
            treatment_confounders = (A = confounders,)
        )
        
        # Standard TMLE
        standard_estimator = Tmle()
        standard_result, _ = standard_estimator(Ψ, dataset; verbosity = 0)
        
        # LASSO CTMLE with manual lambda
        lasso_strategy = LassoCTMLE(
            confounders = confounders, 
            lambda_path = [0.1, 0.01], 
            patience = 2
        )
        lasso_estimator = Tmle(collaborative_strategy = lasso_strategy)
        lasso_result, _ = lasso_estimator(Ψ, dataset; verbosity = 0)
        
        @test !isnan(estimate(standard_result))
        @test !isnan(estimate(lasso_result))
    end
    
    @testset "Automatic lambda generation" begin
        Random.seed!(456)
        n = 150
        p = 8
        W_df = DataFrame(Dict(Symbol("W$i") => randn(n) for i in 1:p))
        A_vals = rand([0, 1], n)
        A = categorical(A_vals; levels=[0, 1])
        Y = 2.0 * A_vals + sum(Matrix(W_df[:, 1:3]), dims=2)[:, 1] + randn(n) * 0.3
        dataset = hcat(W_df, DataFrame(A=A, Y=Y))
        
        confounders = [Symbol("W$i") for i in 1:p]
        Ψ = ATE(
            outcome = :Y,
            treatment_values = (A = (case = 1, control = 0),),
            treatment_confounders = (A = confounders,)
        )
        
        auto_strategy = LassoCTMLE(confounders = confounders, patience = 3)
        
        @test length(auto_strategy.lambda_path) == 0
        @test auto_strategy.current_iteration == 0
        
        auto_estimator = Tmle(collaborative_strategy = auto_strategy)
        auto_result, _ = auto_estimator(Ψ, dataset; verbosity = 0)
        
        @test !isnan(estimate(auto_result))
        
        standard_estimator = Tmle()
        standard_result, _ = standard_estimator(Ψ, dataset; verbosity = 0)
        
        @test estimate(auto_result) != estimate(standard_result)
    end
    
    
end
