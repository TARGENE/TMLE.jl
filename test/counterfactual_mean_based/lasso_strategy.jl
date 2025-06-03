module TestLassoCTMLE

using Test
using TMLE
using DataFrames
using Random

Random.seed!(1234)

@testset "LassoCTMLE Interface" begin
    n = 200
    W1 = randn(n)
    W2 = randn(n)
    A = rand(Bool, n)
    Y = 2 .+ 1.5 .* A .+ 0.5 .* W1 .- 0.3 .* W2 .+ randn(n)
    df = DataFrame(Y=Y, A=A, W1=W1, W2=W2)

    Ψ = TMLE.AIE(
        outcome = :Y,
        treatment_values = (A=(case=1, control=0),),
        treatment_confounders = (A=[:W1, :W2],)
    )

    lasso_strategy = LassoCTMLE(
        Vector{Float64}(), 0, Vector{Any}(), Float64[], 0, nothing, Float64[], Float64[], Float64[], nothing, Vector{Dict}() 
    )

    TMLE.initialise!(lasso_strategy, Ψ)
    while !TMLE.exhausted(lasso_strategy)
        TMLE.update!(lasso_strategy, nothing, df)
    end
    result = TMLE.finalise!(lasso_strategy)

    @test typeof(result) == NamedTuple
    @test haskey(result, :ATE)
    @test haskey(result, :SE)
    @test !isnan(result.ATE)
    @test !isnan(result.SE)
    @test result.SE > 0

    @info "LassoCTMLE ATE estimate: $(result.ATE), SE: $(result.SE)"
end

end 
