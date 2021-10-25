module TestFluctuations

using Test
using TMLE

@testset "Test continuousfluctuation" begin
    query = (t₁=[false, true], t₂=["a", "b"])
    F = continuousfluctuation(query=query)
    @test F.glm.fit_intercept == false
    @test F.glm.offsetcol == :offset
    @test F.query == query
    @test F.indicators == Dict(
        (t₁ = 1, t₂ = "a") => -1,
        (t₁ = 0, t₂ = "b") => -1,
        (t₁ = 0, t₂ = "a") => 1,
        (t₁ = 1, t₂ = "b") => 1
    )

end

@testset "Test BinaryFluctuation" begin
    F = binaryfluctuation(query=nothing)
    @test F.glm.fit_intercept == false
    @test F.glm.offsetcol == :offset
    @test F.query === nothing
    @test F.indicators === nothing
end

@testset "Test setproperty!(query) sets indicator functions" begin
    query = (t₁=[true, false],)
    F = binaryfluctuation(query=query)

    @test F.indicators == Dict(
        (t₁ = 1,) => 1,
        (t₁ = 0,) => -1
    )

    F.query = (t₁=[false, true], t₂=["a", "b"])

    @test F.query == (t₁=[false, true], t₂=["a", "b"])
    @test F.indicators == Dict(
        (t₁ = 1, t₂ = "a") => -1,
        (t₁ = 0, t₂ = "b") => -1,
        (t₁ = 0, t₂ = "a") => 1,
        (t₁ = 1, t₂ = "b") => 1
    )

    @test_throws ArgumentError F.indicators = Dict()
end

end;

true