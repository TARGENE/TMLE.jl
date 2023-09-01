module NonRegressionTest

using Test
using TMLE
using CSV
using DataFrames 
using CategoricalArrays
using MLJGLMInterface

@testset "Test ATE on perinatal dataset." begin
    # This is a non-regression test which was checked against the R tmle3 package
    dataset = CSV.read(joinpath("data", "perinatal.csv"), DataFrame, missingstring=["", "NA"])
    confounders = [:apgar1, :apgar5, :gagebrth, :mage, :meducyrs, :sexn]
    dataset.haz01 = categorical(dataset.haz01)
    dataset.parity01 = categorical(dataset.parity01)
    for col in confounders
        dataset[!, col] = float(dataset[!, col])
    end

    Ψ = ATE(
        outcome=:haz01, 
        treatment=(parity01=(case=1, control=0),),
        confounders=confounders,
        outcome_model = TreatmentTransformer() |> LinearBinaryClassifier(),
        treatment_model = LinearBinaryClassifier()
    )

    tmle_result, targeted_factors = tmle!(Ψ, dataset; ps_lowerbound=0.025, verbosity=0)
    # TMLE
    @test estimate(tmle_result) ≈ -0.185533 atol = 1e-6
    @test naive_plugin_estimate!(Ψ, dataset) ≈ -0.150078 atol = 1e-6
    l, u = confint(OneSampleTTest(tmle_result))
    @test l ≈ -0.279246 atol = 1e-6
    @test u ≈ -0.091821 atol = 1e-6
    @test OneSampleZTest(tmle_result) isa OneSampleZTest

end

end

true