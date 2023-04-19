module TestConfigurations

using Test
using TMLE
using Serialization

@testset "Test configurations" begin
    param_file = "parameters_sample.yaml"
    parameters = [
        IATE(;
            target=:Y1, 
            treatment=(T2 = (case = 1, control = 0), T1 = (case = 1, control = 0)), 
            confounders=[:W1, :W2], 
            covariates=Symbol[:C1]
        ),
        ATE(;
            target=:Y3, 
            treatment=(T3 = (case = 1, control = 0), T1 = (case = "AC", control = "CC")), 
            confounders=[:W1], 
            covariates=Symbol[]
        ),
        CM(;target=:Y3, treatment=(T3 = "AC", T1 = "CC"), confounders=[:W1], covariates=Symbol[])
    ]
    # Test YAML Serialization
    parameters_to_yaml(param_file, parameters)
    new_parameters = parameters_from_yaml(param_file)
    @test new_parameters == parameters
    rm(param_file)
    # Test basic Serialization
    serialize(param_file, parameters)
    new_parameters = deserialize(param_file)
    @test new_parameters == parameters
    rm(param_file)
end

end

true