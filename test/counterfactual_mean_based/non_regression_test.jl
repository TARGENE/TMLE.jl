module NonRegressionTest

using Test
using TMLE
using CSV
using DataFrames 
using CategoricalArrays
using MLJGLMInterface
using MLJBase
using JSON
using YAML

function regression_tests(tmle_result)
    @test estimate(tmle_result) ≈ -0.185533 atol = 1e-6
    l, u = confint(significance_test(tmle_result))
    @test l ≈ -0.279246 atol = 1e-6
    @test u ≈ -0.091821 atol = 1e-6
    @test OneSampleZTest(tmle_result) isa OneSampleZTest
end

@testset "Test ATE on perinatal dataset." begin
    # This is a non-regression test which was checked against the R tmle3 package
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

    Ψ = ATE(
        outcome=:haz01, 
        treatment_values=(parity01=(case=1, control=0),),
        treatment_confounders=(parity01=confounders,)
    )

    resampling=nothing # No CV
    ps_lowerbound = 0.025 # Cutoff hardcoded in tmle3
    weighted = false # Unweighted fluctuation
    verbosity = 0 # No logs
    max_iter = 1 # One iteration
    tol = nothing # Default tolerance
    tmle = TMLEE(;
        resampling=resampling,
        ps_lowerbound=ps_lowerbound,
        max_iter=max_iter,
        tol=tol,
        weighted=weighted
    )
    
    tmle_result, cache = tmle(Ψ, dataset; verbosity=verbosity);
    regression_tests(tmle_result)
    if VERSION >= v"1.9"
        jsonfile = mktemp()[1]
        TMLE.write_json(jsonfile, [tmle_result])
        results_from_json = TMLE.read_json(jsonfile, use_mmap=false)
        regression_tests(results_from_json[1])

        yamlfile = mktemp()[1]
        TMLE.write_yaml(yamlfile, [emptyIC(tmle_result)])
        results_from_yaml = TMLE.read_yaml(yamlfile)
        regression_tests(results_from_yaml[1])
    end
    # Naive
    naive = NAIVE(with_encoder(LinearBinaryClassifier()))
    naive_result, cache = naive(Ψ, dataset; cache=cache, verbosity=verbosity)
    @test naive_result ≈ -0.150078 atol = 1e-6
end

end

true