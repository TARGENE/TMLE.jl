module TestReport

using Test
using TMLE
using MLJ
using StableRNGs
using HypothesisTests


@testset "Test summary" begin
    query = Query((t=0,), (t=1,))
    r1 = TMLE.Report("y", query, [1, 2, 3, 4], 1, 0.8)
    s1 = briefreport(r1)
    @test s1.query == query
    @test s1.pvalue ≈ 5.88e-8 atol=1e-2
    @test s1.confint[1] ≈ 2.234 atol=1e-2
    @test s1.confint[2] ≈ 4.765 atol=1e-2
    @test s1.estimate == 1.0
    @test s1.initial_estimate == 0.8
    @test s1.stderror ≈ 0.645 atol=1e-2
    @test s1.mean_inf_curve == 2.5

end

@testset "Test Misc" begin
    rng = StableRNG(123)
    n = 100
    T = (t₁=categorical(rand(rng, ["CG", "CC"], n)),
         t₂=categorical(rand(rng, ["AT", "AA"], n)))
    W = (w₁=rand(rng, n), w₂=rand(rng, n))
    y = (y₁=categorical(rand(rng, [true, false], n)),
        y₂=categorical(rand(rng, [true, false], n)))

    queries = [
        Query(case=(t₁="CC", t₂="AT"), control=(t₁="CG", t₂="AA"), name="Query1"),
        Query(case=(t₁="CG", t₂="AT"), control=(t₁="CC", t₂="AA"), name="Query2")
    ]
    Q̅ = ConstantClassifier()
    G = FullCategoricalJoint(ConstantClassifier())
    tmle = TMLEstimator(Q̅, G, queries...)
    
    mach = machine(tmle, T, W, y)
    fit!(mach, verbosity=0)
    # queryreportname
    @test TMLE.queryreportname(30, 43) == :target_30_query_43
    # briefreport
    bfs = briefreport(mach)
    @test sort(collect((bf.target_name, bf.query.name) for bf in bfs)) ==
        [(:y₁, "Query1"),
        (:y₁, "Query2"),
        (:y₂, "Query1"),
        (:y₂, "Query2")]
    
    # queryreport
    qr_id_to_names = Dict(
        (1, 1) => (:y₁, "Query1"),
        (1, 2) => (:y₁, "Query2"),
        (2, 1) => (:y₂, "Query1"),
        (2, 2) => (:y₂, "Query2")
    )
    for target_idx in 1:2
        for query_idx in 1:2
            qr = queryreport(mach, target_idx, query_idx)
            tn, qn = qr_id_to_names[(target_idx, query_idx)]
            @test qr.target_name == tn
            @test qr.query.name == qn
        end
    end

    # z-test
    ztest_results = ztest(mach)

    ztest_result_1 = ztest_results[1]
    @test ztest_result_1.target_name == :y₁
    @test ztest_result_1.query_name == "Query1"
    @test ztest_result_1.test_result isa OneSampleZTest
    @test ztest_result_1.test_result == ztest(mach, 1, 1)
    
    ztest_result_2 = ztest_results[2]
    @test ztest_result_2.target_name == :y₁
    @test ztest_result_2.query_name == "Query2"
    @test ztest_result_2.test_result isa OneSampleZTest
    @test ztest_result_2.test_result == ztest(mach, 1, 2)

    ztest_result_3 = ztest_results[3]
    @test ztest_result_3.target_name == :y₂
    @test ztest_results[3].query_name == "Query1"
    @test ztest_result_3.test_result isa OneSampleZTest
    @test ztest_result_3.test_result == ztest(mach, 2, 1)

    ztest_result_4 = ztest_results[4]
    @test ztest_result_4.target_name == :y₂
    @test ztest_result_4.query_name == "Query2"
    @test ztest_result_4.test_result isa OneSampleZTest
    @test ztest_result_4.test_result == ztest(mach, 2, 2)

end

end

true