using Test
using TMLE
using CSV
using DataFrames
using CategoricalArrays

function non_regression_dataset()
    dataset = CSV.read(
        joinpath(dirname(dirname(pathof(TMLE))), "test", "data", "perinatal.csv"), 
        DataFrame, 
        missingstring=["", "NA"]
    )
    confounders = [:apgar1, :apgar5, :gagebrth, :mage, :meducyrs, :sexn]
    dataset.haz01 = categorical(dataset.haz01)
    dataset.parity01 = categorical(dataset.parity01, ordered=true)
    for col in confounders
        dataset[!, col] = float(dataset[!, col])
    end
    return dataset, confounders
end

@testset "LassoCTMLE on perinatal dataset" begin
    dataset, confounders = non_regression_dataset()
    Ψ = ATE(
        outcome=:haz01,
        treatment_values=(parity01=(case=1, control=0),),
        treatment_confounders=(parity01=confounders,)
    )
    lasso_estimator = Tmle(
        collaborative_strategy=LassoCTMLE(
            confounders=confounders
        )
    )
    lasso_result, _ = lasso_estimator(Ψ, dataset; verbosity=0)
    @test !isnan(estimate(lasso_result))
    @info "LassoCTMLE estimate on perinatal data:" estimate(lasso_result)
end