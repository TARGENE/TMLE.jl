module NonRegressionTest

using Test
using TMLE
using CSV
using DataFrames 
using CategoricalArrays
using MLJGLMInterface
using MLJBase


@testset "Test ATE on perinatal dataset." begin
    # This is a non-regression test which was checked against the R tmle3 package
    dataset = CSV.read(joinpath("data", "perinatal.csv"), DataFrame, missingstring=["", "NA"])
    confounders = [:apgar1, :apgar5, :gagebrth, :mage, :meducyrs, :sexn]
    dataset.haz01 = categorical(dataset.haz01)
    dataset.parity01 = categorical(dataset.parity01, ordered=true)
    for col in confounders
        dataset[!, col] = float(dataset[!, col])
    end

    Ψ = ATE(
        outcome=:haz01, 
        treatment_values=(parity01=(case=1, control=0),),
        treatment_confounders=(parity01=confounders,)
    )
    models = (
        haz01 = with_encoder(LinearBinaryClassifier()),
        parity01 = LinearBinaryClassifier()
        )
    resampling=nothing # No CV
    ps_lowerbound = 0.025 # Cutoff hardcoded in tmle3
    weighted = false # Unweighted fluctuation
    verbosity = 1 # No logs
    tmle = TMLEE(models;
        resampling=resampling,
        ps_lowerbound=ps_lowerbound,
        weighted=weighted
    )
    
    tmle_result, cache = tmle(Ψ, dataset; verbosity=verbosity);
    tmle_result
    tmle_result, cache = tmle(Ψ, dataset; cache=cache, verbosity=verbosity);
    cache
    tmle.models = (
        haz01 = with_encoder(LinearBinaryClassifier()),
        parity01 = LinearBinaryClassifier(fit_intercept=false)
        )
    @test estimate(tmle_result) ≈ -0.185533 atol = 1e-6
    l, u = confint(OneSampleTTest(tmle_result))
    @test l ≈ -0.279246 atol = 1e-6
    @test u ≈ -0.091821 atol = 1e-6
    @test OneSampleZTest(tmle_result) isa OneSampleZTest

    # OSE
    ose = OSE(models; 
        resampling=resampling,
        ps_lowerbound=ps_lowerbound
    )
    ose_result, targeted_factors = ose(Ψ, dataset; verbosity=verbosity)
    ose_result

    # CV-TMLE
    tmle.resampling = StratifiedCV(nfolds=10)
    cv_tmle_result, targeted_factors = tmle(Ψ, dataset; verbosity=verbosity)
    cv_tmle_result

    # Naive
    naive = NAIVE(models.haz01)
    @test naive(Ψ, dataset) ≈ -0.150078 atol = 1e-6

end

end

true