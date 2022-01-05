module TestReport

using Test
using TMLE
using MLJ

@testset "Test standarderror" begin
    @test TMLE.standarderror([1, 2, 3, 4]) ≈ 0.645497 atol=1e-5
    @test TMLE.standarderror([5, 6, 7, 8]) ≈ 0.645497 atol=1e-5
end

@testset "Test influencecurve" begin
    @test TMLE.influencecurve([1, 1, 1], [1, 0, 1], [0.8, 0.1, 0.8], [0.8, 0.2, 0.8], 1) == 
        [0.0
        -0.9
        0.0]

end

@testset "Test summary" begin
    r1 = TMLE.QueryReport((t=[0, 1],), [1, 2, 3, 4], 1, 0.8)
    s1 = TMLE.summary(r1)
    @test s1 == (pvalue = 0.1213352503584821,
                confint = (-0.2651745597610895, 2.2651745597610895),
                estimate = 1.0,
                stderror = 0.6454972243679028,
                initial_estimate = 0.8,
                mean_inf_curve = 2.5,)

    r2 = TMLE.QueryReport((t=["a", "b"],), [5, 6, 7, 8], 0.8, 1)
    s2 = TMLE.summary(r2)
    @test s2 == (pvalue = 0.2152141805194588,
                confint = (-0.46517455976108946, 2.0651745597610898),
                estimate = 0.8,
                stderror = 0.6454972243679028,
                initial_estimate = 1.0,
                mean_inf_curve = 6.5,)

end
end

true