module TestFluctuations

using Test
using TMLE

@testset "Test ContinuousFluctuation" begin
    F = ContinuousFluctuation()
    @test F.glm.fit_intercept == false
    @test F.glm.offsetcol == :offset
end

@testset "Test BinaryFluctuation" begin
    F = BinaryFluctuation()
    @test F.glm.fit_intercept == false
    @test F.glm.offsetcol == :offset
end

end;

true
