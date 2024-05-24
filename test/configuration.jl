module TestConfiguration

using Test
using TMLE
using YAML
using JSON
using Serialization

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

@testset "Test Configurations" begin
    yamlfilename = mktemp()[1]
    jsonfilename = mktemp()[1]
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
    TMLE.write_yaml(yamlfilename, estimands)
    estimands_from_yaml = TMLE.read_yaml(yamlfilename)
    @test estimands_from_yaml == estimands

    TMLE.write_json(jsonfilename, estimands)
    estimands_from_json = TMLE.read_json(jsonfilename, use_mmap=false)
    @test estimands_from_json == estimands
   
    configuration = Configuration(
        estimands=estimands
    )
    config_dict = TMLE.to_dict(configuration)
    TMLE.write_yaml(yamlfilename, configuration)
    TMLE.write_json(jsonfilename, configuration)
    config_from_yaml = TMLE.read_yaml(yamlfilename)
    config_from_json = TMLE.read_json(jsonfilename, use_mmap=false)
    for loaded_config in (config_from_yaml, config_from_json)
        @test loaded_config.scm === nothing
        @test loaded_config.adjustment === nothing
        reconstructed_estimands = loaded_config.estimands
        @test reconstructed_estimands == estimands
    end

end

@testset "Test with an SCM and causal estimands" begin
    yamlfilename = mktemp()[1]
    jsonfilename = mktemp()[1]
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
    TMLE.write_yaml(yamlfilename, configuration)
    TMLE.write_json(jsonfilename, configuration)

    config_from_yaml = TMLE.read_yaml(yamlfilename)
    config_from_json = TMLE.read_json(jsonfilename, use_mmap=false)

    for loaded_config in (config_from_yaml, config_from_json)
        scm = loaded_config.scm
        adjustment = loaded_config.adjustment
        # Estimand 1
        estimand₁ = loaded_config.estimands[1]
        @test estimand₁ == estimands[1]
        statistical_estimand₁ = identify(adjustment, estimand₁, scm)
        @test identify(adjustment, estimand₁, scm) == identify(configuration.adjustment, estimands[1], configuration.scm)
        # Estimand 2
        estimand₂ = loaded_config.estimands[2]
        @test estimand₂ == estimands[2]
    end
end

@testset "Test serialization" begin
    ATE₁ = ATE(
        outcome=:Y,
        treatment_values = (T=(case=1, control=0),),
        treatment_confounders = (T=[:W],)
    )
    ATE₂ = ATE(
        outcome=:Y,
        treatment_values = (T=(case=2, control=1),),
        treatment_confounders = (T=[:W],)
    )
    diff = JointEstimand(-, (ATE₁, ATE₂))
    estimands = [ATE₁, ATE₂, diff]
    jlsfile = mktemp()[1]
    serialize(jlsfile, estimands)
    estimands_from_jls = deserialize(jlsfile)
    @test estimands_from_jls[1] == estimands[1]
    @test estimands_from_jls[2] == estimands[2]
    @test estimands_from_jls[3] == estimands[3]
end


end

true