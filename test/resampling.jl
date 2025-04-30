module TestResampling

using Test
using TMLE
using CategoricalArrays
using MLJBase
using StableRNGs
using DataFrames

@testset "Test CausalStratifiedCV" begin
    dataset = DataFrame(
        T1 = categorical([0, 1, 0, 0, 0, 1, missing]),
        T2 = categorical([0, 1, 1, 0, 0, 1, 0]),
        W = [0, 0, 1, 0, 1, 0, missing],
        Y_bin = categorical([1, 0, 1, 1, 0, 1, 1]),
        Y_cont = rand(7),
    )
    resampling = CausalStratifiedCV(resampling=StratifiedCV(nfolds=2))
    # Binary outcome, 1 treatment
    Ψ = ATE(
        outcome=:Y_bin,
        treatment_values=(T1=(case=1, control=0),),
    )
    MLJBase.fit!(resampling, Ψ, dataset)
    @test resampling.treatment_variables == [:T1]
    X = dataset[!, Not(:Y_bin)]
    y = dataset.Y_bin
    
    stratification_col = fill("", nrows(X))

    TMLE.aggregate_features!(stratification_col, resampling.treatment_variables, X)
    @test stratification_col == ["0_", "1_", "0_", "0_", "0_", "1_", "missing_"]
    TMLE.update_stratification_col_if_finite!(stratification_col, y)
    @test stratification_col == ["0_1_", "1_0_", "0_1_", "0_1_", "0_0_", "1_1_", "missing_1_"]
    train_validation_indices = TMLE.get_train_validation_indices(resampling, Ψ, dataset)
    @test length(train_validation_indices) == 2
    
    # Continuous outcome, 2 treatments
    Ψ = ATE(
        outcome=:Y_cont,
        treatment_values=(T1=(case=1, control=0), T2=(case=1, control=0)),
    )
    MLJBase.fit!(resampling, Ψ, dataset)
    @test resampling.treatment_variables == [:T1, :T2]
    X = dataset[!, Not(:Y_cont)]
    y = dataset.Y_cont
    stratification_col = fill("", nrows(X))
    TMLE.aggregate_features!(stratification_col, resampling.treatment_variables, X)
    @test stratification_col == ["0_0_", "1_1_", "0_1_", "0_0_", "0_0_", "1_1_", "missing_0_"]
    # continuous y does not change the stratification
    TMLE.update_stratification_col_if_finite!(stratification_col, y)
    @test stratification_col == ["0_0_", "1_1_", "0_1_", "0_0_", "0_0_", "1_1_", "missing_0_"]
    train_validation_indices = TMLE.get_train_validation_indices(resampling, Ψ, dataset)
    @test length(train_validation_indices) == 2

end

end

true