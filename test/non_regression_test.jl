module NonRegressionTest

using Test
using TMLE
using MLJLinearModels
using CSV
using DataFrames 
using CategoricalArrays
using MLJGLMInterface


@testset "Test ATE on perinatal dataset." begin
    # This is a non-regression test which was checked against the R tmle3 package
    data = CSV.read(joinpath("data", "perinatal.csv"), DataFrame, missingstring=["", "NA"])
    confounders = [:apgar1, :apgar5, :gagebrth, :mage, :meducyrs, :sexn]
    data.haz01 = categorical(data.haz01)
    data.parity01 = categorical(data.parity01)
    for col in confounders
        data[!, col] = float(data[!, col])
    end
    Ψ = ATE(
        target=:haz01,
        treatment = (parity01=(case=1, control=0),),
        confounders = confounders
    )
    η_spec = NuisanceSpec(
        LinearBinaryClassifier(),
        LinearBinaryClassifier()
    )
    r, cache = tmle(Ψ, η_spec, data, threshold=0.025)
    l, u = confint(OneSampleTTest(r.tmle))
    @test TMLE.estimate(r.tmle) ≈ -0.185533 atol = 1e-6
    @test l ≈ -0.279246 atol = 1e-6
    @test u ≈ -0.091821 atol = 1e-6
end

end

true