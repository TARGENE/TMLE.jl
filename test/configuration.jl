module TestConfiguration

using Test
using TMLE
using YAML

@testset "Test Configurations" begin
    outfilename = "test_configuration.yml"
    estimands = [
        IATE(
            outcome=:Y1, 
            treatment_values= (
                T1 = (case = 1, control = 0), 
                T2 = (case = 1, control = 0)
            ), 
            treatment_confounders=(
                T1 = [:W11, :W12],
                T2 = [:W21, :W22],
            ),
            outcome_extra_covariates=[:C1]
        ),
        ATE(;
            outcome=:Y3, 
            treatment_values = (
                T1 = (case = 1, control = 0), 
                T3 = (case = "AC", control = "CC")
            ), 
            treatment_confounders=(
                T1 = [:W], 
                T3 = [:W]
            ), 
        ),
        CM(;
            outcome=:Y3, 
            treatment_values=(T3 = "AC", T1 = "CC"), 
            treatment_confounders=(T3 = [:W], T1 = [:W])
        )
    ]
    configuration = Configuration(
        estimands=estimands
    )
    config_dict = to_dict(configuration)
    configuration_to_yaml(outfilename, configuration)
    config_from_yaml = configuration_from_yaml(outfilename)
    @test config_from_yaml.scm === nothing
    @test config_from_yaml.adjustment === nothing
    for index in 1:length(estimands)
        estimand = configuration.estimands[index]
        reconstructed_estimand = config_from_yaml.estimands[index]
        @test estimand.outcome == reconstructed_estimand.outcome
        for treatment in keys(estimand.treatment_values)
            @test estimand.treatment_values[treatment] == reconstructed_estimand.treatment_values[treatment]
            @test estimand.treatment_confounders[treatment] == reconstructed_estimand.treatment_confounders[treatment]
            @test estimand.outcome_extra_covariates == reconstructed_estimand.outcome_extra_covariates
        end
    end
    rm(outfilename)

    # With a StaticSCM, some Causal estimands and an Adjustment Method
    estimands = [
        IATE(
            outcome=:Y1, 
            treatment_values= (
                T1 = (case = 1, control = 0), 
                T2 = (case = 1, control = 0)
            )
        ),
        ATE(;
            outcome=:Y3, 
            treatment_values = (
                T1 = (case = 1, control = 0), 
                T3 = (case = "AC", control = "CC")
            ), 
            treatment_confounders=(
                T1 = [:W], 
                T3 = [:W]
            ), 
        ),
    ]

    configuration = Configuration(
        estimands=estimands,
        scm = StaticSCM(
            confounders = [:W],
            outcomes = [:Y1],
            treatments = [:T1, :T2]),
        adjustment=BackdoorAdjustment([:C1, :C2])
    )   
    configuration_to_yaml(outfilename, configuration)
    config_from_yaml = configuration_from_yaml(outfilename)
    scm = config_from_yaml.scm
    adjustment = config_from_yaml.adjustment
    # Estimand 1
    estimand₁ = config_from_yaml.estimands[1]
    statistical_estimand₁ = identify(adjustment, estimand₁, scm)
    @test statistical_estimand₁.outcome == :Y1
    @test statistical_estimand₁.treatment_values.T1 == (case=1, control=0)
    @test statistical_estimand₁.treatment_values.T2 == (case=1, control=0)
    @test statistical_estimand₁.treatment_confounders.T2 == (:W,)
    @test statistical_estimand₁.treatment_confounders.T1 == (:W,)
    @test statistical_estimand₁.outcome_extra_covariates == (:C1, :C2)
    # Estimand 2
    estimand₂ = config_from_yaml.estimands[2]
    @test estimand₂.outcome == :Y3
    @test estimand₂.treatment_values.T1 == (case = 1, control = 0)
    @test estimand₂.treatment_values.T3 == (case = "AC", control = "CC")
    @test estimand₂.treatment_confounders.T1 == (:W,)
    @test estimand₂.treatment_confounders.T3 == (:W,)

    rm(outfilename)
end

end

true