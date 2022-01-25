module TestReport

using Test
using TMLE
using MLJ

@testset "Test influencecurve" begin
    @test TMLE.influencecurve([1, 1, 1], [1, 0, 1], [0.8, 0.1, 0.8], [0.8, 0.2, 0.8], 1) == 
        [0.0
        -0.9
        0.0]
end

@testset "Test summary" begin
    query = Query((t=0,), (t=1,))
    r1 = TMLE.QueryReport(query, [1, 2, 3, 4], 1, 0.8)
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
end

true