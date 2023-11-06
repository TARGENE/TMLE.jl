module TestAdjustment

using Test
using TMLE

@testset "Test BackdoorAdjustment" begin
    scm = StaticSCM(
        outcomes = [:Y₁, :Y₂],
        treatments = [:T₁, :T₂],
        confounders = [:W₁, :W₂],
    )
    causal_estimands = [
        CM(outcome=:Y₁, treatment_values=(T₁=1,)),
        ATE(outcome=:Y₁, treatment_values=(T₁=(case=1, control=0),)),
        ATE(outcome=:Y₁, treatment_values=(T₁=(case=1, control=0), T₂=(case=1, control=0))),
        IATE(outcome=:Y₁, treatment_values=(T₁=(case=1, control=0), T₂=(case=1, control=0))),
    ]
    method = BackdoorAdjustment(outcome_extra_covariates=[:C])
    statistical_estimands = [identify(method, estimand, scm) for estimand in causal_estimands]

    for (causal_estimand, statistical_estimand) in zip(causal_estimands, statistical_estimands)
        @test statistical_estimand.outcome == causal_estimand.outcome
        @test statistical_estimand.treatment_values == causal_estimand.treatment_values
        @test statistical_estimand.outcome_extra_covariates == (:C,)
    end
    @test statistical_estimands[1].treatment_confounders == (T₁=(:W₁, :W₂),)
    @test statistical_estimands[2].treatment_confounders == (T₁=(:W₁, :W₂),)
    @test statistical_estimands[3].treatment_confounders == (T₁=(:W₁, :W₂), T₂=(:W₁, :W₂))
    @test statistical_estimands[4].treatment_confounders == (T₁=(:W₁, :W₂), T₂=(:W₁, :W₂))
end

@testset "Test to_dict" begin
    adjustment = BackdoorAdjustment(outcome_extra_covariates=[:C])
    adjustment_dict = to_dict(adjustment)
    @test adjustment_dict == Dict(
        :outcome_extra_covariates => [:C],
        :type                     => "BackdoorAdjustment"
    )
    @test from_dict!(adjustment_dict) == adjustment
end

end

true