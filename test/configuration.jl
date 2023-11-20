module TestConfiguration

using Test
using TMLE
using YAML
using JSON

function check_estimands(reconstructed_estimands, estimands)
    for (index, estimand) in enumerate(estimands)
        reconstructed_estimand = reconstructed_estimands[index]
        @test estimand.outcome == reconstructed_estimand.outcome
        for treatment in keys(estimand.treatment_values)
            @test estimand.treatment_values[treatment] == reconstructed_estimand.treatment_values[treatment]
            @test estimand.treatment_confounders[treatment] == reconstructed_estimand.treatment_confounders[treatment]
            @test estimand.outcome_extra_covariates == reconstructed_estimand.outcome_extra_covariates
        end
    end
end

outprefix = "test_serialized."
yamlfilename = string(outprefix, "yaml")
jsonfilename = string(outprefix, "json")

@testset "Test Configurations" begin
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
    write_yaml(yamlfilename, estimands)
    estimands_from_yaml = read_yaml(yamlfilename)
    check_estimands(estimands_from_yaml, estimands)
    rm(yamlfilename)
    write_json(jsonfilename, estimands)
    estimands_from_json = read_json(jsonfilename)
    check_estimands(estimands_from_json, estimands)
    rm(jsonfilename)
   
    configuration = Configuration(
        estimands=estimands
    )
    config_dict = to_dict(configuration)
    write_yaml(yamlfilename, configuration)
    write_json(jsonfilename, configuration)
    config_from_yaml = read_yaml(yamlfilename)
    config_from_json = read_json(jsonfilename)
    for loaded_config in (config_from_yaml, config_from_json)
        @test loaded_config.scm === nothing
        @test loaded_config.adjustment === nothing
        reconstructed_estimands = loaded_config.estimands
        check_estimands(reconstructed_estimands, estimands)
    end
    rm(yamlfilename)
    rm(jsonfilename)
end

@testset "Test with an SCM and causal estimands" begin
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
    write_yaml(yamlfilename, configuration)
    write_json(jsonfilename, configuration)

    config_from_yaml = read_yaml(yamlfilename)
    config_from_json = read_json(jsonfilename)

    for loaded_config in (config_from_yaml, config_from_json)
        scm = loaded_config.scm
        adjustment = loaded_config.adjustment
        # Estimand 1
        estimand₁ = loaded_config.estimands[1]
        statistical_estimand₁ = identify(adjustment, estimand₁, scm)
        @test statistical_estimand₁.outcome == :Y1
        @test statistical_estimand₁.treatment_values.T1 == (case=1, control=0)
        @test statistical_estimand₁.treatment_values.T2 == (case=1, control=0)
        @test statistical_estimand₁.treatment_confounders.T2 == (:W,)
        @test statistical_estimand₁.treatment_confounders.T1 == (:W,)
        @test statistical_estimand₁.outcome_extra_covariates == (:C1, :C2)
        # Estimand 2
        estimand₂ = loaded_config.estimands[2]
        @test estimand₂.outcome == :Y3
        @test estimand₂.treatment_values.T1 == (case = 1, control = 0)
        @test estimand₂.treatment_values.T3 == (case = "AC", control = "CC")
        @test estimand₂.treatment_confounders.T1 == (:W,)
        @test estimand₂.treatment_confounders.T3 == (:W,)
    end
    rm(yamlfilename)
    rm(jsonfilename)
end

end

true