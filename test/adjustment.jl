module TestAdjustment

using Test
using TMLE

@testset "Test BackdoorAdjustment" begin
    scm = SCM(
        SE(:Y, [:I, :V, :C]),
        SE(:V, [:T, :K]),
        SE(:T, [:W, :X]),
        SE(:I, [:W, :G]),
    )
    Ψ = CM(scm, treatment=(T=1,), outcome=:Y)
    adjustment_method = BackdoorAdjustment()
    @test TMLE.get_models_input_variables(adjustment_method, Ψ) == (
        T = [:W, :X],
        Y = [:T, :W, :X]
    )
    adjustment_method = BackdoorAdjustment(outcome_extra=[:C])
    @test TMLE.get_models_input_variables(adjustment_method, Ψ) == (
        T = [:W, :X],
        Y = [:T, :W, :X, :C]
    )
end

end

true