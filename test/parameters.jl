module TestParameters

using Test
using TMLE

@testset "Test variables accessors" begin
    n = 10
    dataset = (
        W₁=rand(n),
        W₂=rand(n),
        C₁=rand(n),
        T₁=rand([1, 0], n),
        y=rand(n),  
        )
    Ψ = CM(
        treatment=(T₁=1,),
        confounders=[:W₁, :W₂],
        target =:y
        )
    @test TMLE.treatments(Ψ) == [:T₁]
    @test TMLE.target(Ψ) == :y
    @test TMLE.treatment_and_confounders(Ψ) == [:W₁, :W₂, :T₁]
    @test TMLE.confounders_and_covariates(Ψ) == [:W₁, :W₂]
    @test TMLE.allcolumns(Ψ) == [:W₁, :W₂, :T₁, :y]

    Ψ = CM(
        treatment=(T₁=1,),
        confounders=[:W₁, :W₂],
        target =:y,
        covariates=[:C₁]
        )
    @test TMLE.treatments(Ψ) == [:T₁]
    @test TMLE.target(Ψ) == :y
    @test TMLE.treatment_and_confounders(Ψ) == [:W₁, :W₂, :T₁]
    @test TMLE.confounders_and_covariates(Ψ) == [:W₁, :W₂, :C₁]
    @test TMLE.allcolumns(Ψ) == [:W₁, :W₂, :C₁, :T₁, :y]
end

end

true