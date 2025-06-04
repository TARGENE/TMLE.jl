using Test
using TMLE  

@testset "LassoCTMLE basic functionality" begin
    
    using DataFrames
    n = 100
    df = DataFrame(
        W1 = randn(n),
        W2 = randn(n),
        A = rand(Bool, n),
        Y = rand(Bool, n)
    )

    strategy = LassoCTMLE(confounders=[:W1, :W2], cv_folds=3)

    Ψ = ATE(treatment=:A, outcome=:Y, confounders=[:W1, :W2])

    result = tmle(strategy, Ψ, df)

    @test result.ψ isa Float64
    @test result.SE > 0
    @test result.CI[1] < result.CI[2]
    @test 0.0 <= result.pvalue <= 1.0
end

