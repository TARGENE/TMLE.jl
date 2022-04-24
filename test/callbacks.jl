module TestCallbacks

using Test
using TMLE
using StableRNGs
using MLJModels
using CategoricalArrays
using JLD2
using MLJBase

@testset "Test JLD2Saver" begin
    rng = StableRNG(123)
    n = 100
    T = (t₁=categorical(rand(rng, ["CG", "CC", "GG"], n)),
         t₂=categorical(rand(rng, ["AT", "AA", "TT"], n)))
    W = (w₁=rand(rng, n), w₂=rand(rng, n))
    Y = (
        y₁=categorical(rand(rng, [0, 1], n)),
        y₂=categorical(rand(rng, [0, 1], n))
    )
    
    query₁ = Query(case=(t₁="CC", t₂="AT"), control=(t₁="CG", t₂="AA"), name="Query_1")
    query₂ = Query(case=(t₁="GG", t₂="AA"), control=(t₁="CG", t₂="TT"), name="Query_2")

    Q̅ = ConstantClassifier()
    G = FullCategoricalJoint(ConstantClassifier())
    outfile = "testresult.jld2"
    tmle = TMLEstimator(Q̅, G, query₁, query₂)

    # Full save
    result = TMLE.fit(tmle, T, W, Y, 
                        verbosity=0, 
                        callbacks=JLD2Saver(outfile, true))

    resultfile = jldopen(outfile)
    
    @test resultfile["low_propensity_scores"] == result.low_propensity_scores

    machines = resultfile["machines"]
    for key in ("Encoder", "G", "Q_1", "Q_2", "F_1_1", "F_1_2", "F_2_1", "F_2_2")
        @test machines[key] isa Machine
    end
    tmlereports = resultfile["tmlereports"]
    for key in ("1_1", "1_2", "2_1", "2_2")
        @test tmlereports[key] isa TMLE.TMLEReport
    end
    rm(outfile)

    # Don't save machines
    result = TMLE.fit(tmle, T, W, Y, 
                        verbosity=0, 
                        callbacks=JLD2Saver(outfile, false))

    resultfile = jldopen(outfile)
    
    @test resultfile["low_propensity_scores"] == result.low_propensity_scores

    @test haskey(resultfile, "machines") == false
    
    tmlereports = resultfile["tmlereports"]
    for key in ("1_1", "1_2", "2_1", "2_2")
        @test tmlereports[key] isa TMLE.TMLEReport
    end
    rm(outfile)
end


end

true