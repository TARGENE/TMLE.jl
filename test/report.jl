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
    r1 = TMLE.QueryReport((t=[0, 1],), [1, 2, 3, 4], 1, 0.8)
    s1 = briefreport(r1)
    @test s1 == (query = (t = [0, 1],),
                pvalue = 5.887764274517263e-8,
                confint = (2.234848688118337, 4.765151311881663),
                estimate = 1.0,
                initial_estimate = 0.8,
                stderror = 0.6454972243679028,
                mean_inf_curve = 2.5,)

end
end

true