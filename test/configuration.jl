module TestConfigurations

using Test
using TMLE
using Serialization

@testset "Test configurations" begin
    param_file = "estimands_sample.yaml"
    estimands = [
        IATE(;
            outcome=:Y1, 
            treatment=(T2 = (case = 1, control = 0), T1 = (case = 1, control = 0)), 
            confounders=[:W1, :W2], 
            covariates=Symbol[:C1]
        ),
        ATE(;
            outcome=:Y3, 
            treatment=(T3 = (case = 1, control = 0), T1 = (case = "AC", control = "CC")), 
            confounders=[:W1], 
            covariates=Symbol[]
        ),
        CM(;outcome=:Y3, treatment=(T3 = "AC", T1 = "CC"), confounders=[:W1], covariates=Symbol[])
    ]
    # Test YAML Serialization
    estimands_to_yaml(param_file, estimands)
    new_estimands = estimands_from_yaml(param_file)
    @test new_estimands == estimands
    rm(param_file)
    # Test basic Serialization
    serialize(param_file, estimands)
    new_estimands = deserialize(param_file)
    @test new_estimands == estimands
    rm(param_file)
end

end

true