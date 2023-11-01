module TestConfigurations

using Test
using TMLE
using Serialization

@testset "Test configurations" begin
    estimand_file = "estimands_sample.yaml"
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
    # Test YAML Serialization
    estimands_to_yaml(estimand_file, estimands)
    loaded_estimands = estimands_from_yaml(estimand_file)
    @test loaded_estimands == estimands
    rm(estimand_file)
    # Test basic Serialization
    serialize(estimand_file, estimands)
    loaded_estimands = deserialize(estimand_file)
    @test loaded_estimands == estimands
    rm(estimand_file)
end

end

true