module TestEstimands

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

@testset "Test optimize_ordering" begin
    parameters = [
        ATE(
            target=:Y, 
            treatment=(T₁=(case=1, control=0), T₂=(case="AC", control="CC")),
            confounders=[:W₁, :W₂]
        ),
        ATE(
            target=:Y, 
            treatment=(T₁=(case=1, control=0),),
            confounders=[:W₁]
        ),
        ATE(
            target=:Y₂, 
            treatment=(T₁=(case=1, control=0), T₂=(case="AC", control="CC")),
            confounders=[:W₁, :W₂],
        ),
        CM(
            target=:Y, 
            treatment=(T₁=0,),
            confounders=[:W₁],
            covariates=[:C₁]
        ),
        IATE(
            target=:Y, 
            treatment=(T₁=(case=1, control=0), T₂=(case="AC", control="CC")),
            confounders=[:W₁, :W₂]
        ),
        CM(
            target=:Y, 
            treatment=(T₁=0,),
            confounders=[:W₁]
        ),
        ATE(
            target=:Y₂, 
            treatment=(T₁=(case=1, control=0),),
            confounders=[:W₁],
            covariates=[:C₁]
        ),
        ATE(
            target=:Y₂, 
            treatment=(T₁=(case=0, control=1), T₂=(case="AC", control="CC")),
            confounders=[:W₁, :W₂],
            covariates=[:C₂]
        ),
    ]
    # Non mutating function
    ordered_parameters = optimize_ordering(parameters)
    expected_ordering = [
        ATE(:Y, (T₁ = (case = 1, control = 0),), [:W₁], Symbol[]),
        CM(:Y, (T₁ = 0,), [:W₁], Symbol[]),
        CM(:Y, (T₁ = 0,), [:W₁], [:C₁]),
        ATE(:Y₂, (T₁ = (case = 1, control = 0),), [:W₁], [:C₁]),
        ATE(:Y, (T₁ = (case = 1, control = 0), T₂ = (case = "AC", control = "CC")), [:W₁, :W₂], Symbol[]),
        IATE(:Y, (T₁ = (case = 1, control = 0), T₂ = (case = "AC", control = "CC")), [:W₁, :W₂], Symbol[]),
        ATE(:Y₂, (T₁ = (case = 1, control = 0), T₂ = (case = "AC", control = "CC")), [:W₁, :W₂], Symbol[]),
        ATE(:Y₂, (T₁ = (case = 0, control = 1), T₂ = (case = "AC", control = "CC")), [:W₁, :W₂], [:C₂])
    ]
    @test ordered_parameters == expected_ordering
    # Mutating function
    optimize_ordering!(parameters)
    @test parameters == ordered_parameters
end

@testset "Test structs are concrete types" begin
    @test isconcretetype(ATE)
    @test isconcretetype(IATE)
    @test isconcretetype(CM)
    @test isconcretetype(TMLE.ALEstimate{Float64})
    @test isconcretetype(TMLE.TMLEResult{ATE, Float64})
    @test isconcretetype(TMLE.TMLEResult{IATE, Float64})
    @test isconcretetype(TMLE.TMLEResult{CM, Float64})
end

end

true